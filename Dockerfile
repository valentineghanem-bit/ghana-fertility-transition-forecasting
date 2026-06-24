# Full computational environment for the Ghana subnational fertility transition study.
# Python 3.12 (primary analysis + forecasting + figures) and R 4.3 (spdep spatial diagnostics).
FROM python:3.12-slim

# R + system libraries for spdep
RUN apt-get update && apt-get install -y --no-install-recommends \
        r-base \
        libgdal-dev libgeos-dev libproj-dev libudunits2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# R spatial-diagnostics dependency (reference implementation in analysis/spatial_diagnostics.R)
RUN R -e "install.packages(c('spdep','sp'), repos='https://cloud.r-project.org')"

COPY . .

# Default: run the full Python analytical pipeline
CMD ["bash", "run_all.sh"]
