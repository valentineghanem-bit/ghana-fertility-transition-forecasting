# spatial_diagnostics.R — Project 15. Reference R implementation of the district spatial diagnostics
# (Global Moran's I + Local Moran's I / LISA), parallel to analysis/spatial_lisa.py.
# Uses spdep with k-nearest-neighbour spatial weights (k = 6) from district centroids, so all 261
# districts enter the analysis (including the 3 with no own GeoJSON polygon). Ghana has 261 districts.
#
# Run:  Rscript analysis/spatial_diagnostics.R
# Requires: R >= 4.3 with spdep, sp.

suppressPackageStartupMessages({
  library(spdep)
})

SRC <- "data/processed/district_cross_section_2022_261.csv"
VARS <- c("poverty_rate", "illiteracy_rate", "uninsured_rate", "under15_share", "women_15_64_share")
K <- 6
set.seed(42)

df <- read.csv(SRC, stringsAsFactors = FALSE)
stopifnot(nrow(df) == 261)

# k-nearest-neighbour spatial weights from centroids (row-standardised), longlat = great-circle
coords <- as.matrix(df[, c("lon", "lat")])
knn <- knearneigh(coords, k = K, longlat = TRUE)
listw <- nb2listw(knn2nb(knn), style = "W")

cat("=== Global Moran's I (k =", K, "KNN; 261 districts; R/spdep) ===\n")
global <- data.frame()
for (v in VARS) {
  mt <- moran.test(df[[v]], listw, randomisation = TRUE, zero.policy = TRUE)
  global <- rbind(global, data.frame(
    variable = v,
    morans_I = round(unname(mt$estimate[1]), 4),
    expectation = round(unname(mt$estimate[2]), 4),
    z = round(unname(mt$statistic), 2),
    p_value = signif(mt$p.value, 3)
  ))
}
print(global, row.names = FALSE)

# Local Moran's I (LISA) for the strongest-clustering covariate (illiteracy)
cat("\n=== LISA cluster counts — illiteracy_rate (p < 0.05) ===\n")
x <- df$illiteracy_rate
lm <- localmoran(x, listw, zero.policy = TRUE)
lag <- lag.listw(listw, x)
sig <- lm[, 5] < 0.05
hi_x <- x > mean(x); hi_lag <- lag > mean(lag)
cluster <- rep("NS", length(x))
cluster[sig &  hi_x &  hi_lag] <- "HH"
cluster[sig & !hi_x & !hi_lag] <- "LL"
cluster[sig &  hi_x & !hi_lag] <- "HL"
cluster[sig & !hi_x &  hi_lag] <- "LH"
print(table(factor(cluster, levels = c("HH", "LL", "HL", "LH", "NS"))))

# persist for cross-checking against the Python pipeline
dir.create("outputs/tables", showWarnings = FALSE, recursive = TRUE)
write.csv(global, "outputs/tables/spatial_global_moran_R.csv", row.names = FALSE)
cat("\nWrote outputs/tables/spatial_global_moran_R.csv (cross-check vs spatial_lisa.py).\n")
