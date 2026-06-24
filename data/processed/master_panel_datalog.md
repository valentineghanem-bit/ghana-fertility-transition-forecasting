# /datalog — Project 15 master panel

Source: DHS StatCompiler subnational exports (Ghana). Frame: 16 regions (2022) x 9 waves.
Harmonisation: analysis/harmonize_regions.py (verified ancestry rule). Extracted 2026-06-22.
Structural missingness = NaN; never imputed before an indicator's first documented wave (Tenet 4).

| indicator | source_csv | source_indicator | unit | waves_present | first_wave | cells_filled | cells_total | inherited_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfr_15_49 | fertility-rates | Total fertility rate 15-49 | children/woman | 9 | 1988 | 144 | 144 | 86 |
| asfr_15_19 | fertility-rates | Age specific fertility rate: 15-19 | births/1000 | 9 | 1988 | 144 | 144 | 88 |
| mcpr_all_women | fp2020 | Current use of any modern method of contraception (all women) | percent | 7 | 1988 | 112 | 144 | 66 |
| mcpr_married | fp2020 | Married women currently using any modern method of contraception | percent | 7 | 1988 | 112 | 144 | 66 |
| any_contra_married | mics-indicators | Married women currently using any method of contraception | percent | 7 | 1988 | 112 | 144 | 66 |
| unmet_need | mics-indicators | Unmet need for family planning | percent | 6 | 1993 | 96 | 144 | 54 |
| edu_secondary_women | select-education-indicators | Women with secondary or higher education | percent | 8 | 1993 | 128 | 144 | 74 |
| age_first_birth | dhs-mobile | Median age at first birth for women age 25-49 | years | 7 | 1988 | 112 | 144 | 66 |
| age_first_marriage | dhs-mobile | Median age at first marriage [Women]: 25-49 | years | 7 | 1988 | 112 | 144 | 66 |
