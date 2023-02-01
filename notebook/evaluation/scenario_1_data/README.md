# TA1 - Scenario 1

## Overall Workflow

1. Searched xDD for relevant models. These can be found in `xdd_query_results`. See [XDD Query Results](#xdd_query_results).
2. Based on the xDD results, MIT and SKEMA produced parameter extractions. See [Text Extractions](#text-extractions).
3. `ACSet` model representation is derived based on prior reading and already existed in TERArium.
4. Perform alignment between `ACSet` states, transitions and parameters to contextualizing metadata that will assist the human modeler in the workbench. See [Model to Metadata Alignment](#model-to-metadata-alignment)

## Decisions Made

* Metadata extrated by TA1 will be provided to TA4 directly so that a user in the HMI workbench has appropriate context to support their modeling task. 
* Example metadata may include things like species/compartment descriptions. E.g. "**Ailing** means symptomatic, infected, and undetected"

## Issues

* Clearly defined format of what is needed by TA2/TA3, in what format, and how it will be used
* TA3 is not apparently directly consuming `py-ACSets` from TA1 at this time; they are ingesting ACSets from TA2
* Need to implement alignment between `py-ACSets` and extracted model assumptions that are neither parameters nor initial conditions. `ACSet` format may be too restrictive so may need work to align it to the `GrometFN`.

### Model to Metadata Alignment
1. Code comments explicitly describe that `alpha` is `death rate`
2. Code variable `alpha` -> to paper variable name based on embedding similarity
3. A human in the loop can assist in annotating the meaning of `alpha`

### xdd_query_reults

 JSON files that contain the results of the information seeking queries for scenario 1. This will be the basis for the text extraction pipelines for MIT and Arizona.

### commented_sources

Python sources originally provided by MITRE and augmented with source code comments for mention linking by the SKEMA team

### Text Extractions
- `scenario1_mit_extractions.xlsx`: Parameter extractions by MIT's reading pipeline
- `scenario1_skema_extractions.xlsx`: Parameter extractions by SKEMA's reading pipeline

### Contact matrices

1. Search result for "SIR age" from xDD leads to https://www.nature.com/articles/s41598-021-94609-3 

2. Figure 1 comes from data in reference 15, "Prem, K. & Cook, A. R. Projecting social contact matrices in 152 countries using contact surveys and demographic data. PLoS Comput. Biol. 13, 20 (2017)." (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697). Figure 1 was converted into a CSV at figure1-10.1038-s41598-021-94609-3.csv. This article was independently found as one of the top hits via a direct xDD query (https://xdd.wisc.edu/api/articles?term=contact%20matrix%20population%20country&match=true&max=10&include_highlights=true)

3. The reference included additional supporting information: 
https://doi.org/10.1371/journal.pcbi.1005697.s001 PDF explaining the data.
https://doi.org/10.1371/journal.pcbi.1005697.s002 Zip file of multiple Excel spreadsheets

4. The Excel spreadsheets include, for ~150 countries, result of contact surveys.  For survey participants, the average number of individuals in each age bin (each age bin is 5 years, up to X=16) reported having contact with. The surveys additionally broke down data by contact location (home, school, work, other). The supporting PDF suggested reweighting the different location datasets to simulate interventions (e.g., reweight to 0 for school contacts to represent school closer, reweight to 0.5 the work and other interactions to represent social distancing).

5. We have uploaded data from USA (for all locations), India (as an example country with
   multi-generational contact) and Belgium (as an example country with no multi-generational contact). These are in files belgium_all_locations_cm.csv, usa_*_cm.csv and india_all_locations_cm.csv.

6. We stopped here.  Note that potentially the weights could be played with for the UK dataset to see how close one can get to the UK during-pandemic surveys. UK datasets from the "Prem, K and Cook A" paper are at uk_*_cm.csv

### Contact Matrix from UK

1. h2020_cm_imputed.csv is from https://github.com/jarvisc1/comix_covid-19-first_wave and is used in
the following paper https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-020-01597-8

### Other datasets

1. Searched for "vaccination" in Terarium found a link to
   https://data.cdc.gov/Vaccinations/Archive-COVID-19-Vaccination-and-Case-Trends-by-Ag/gxj9-t96f/data
  and converted this to usa-vaccination-case-by-age-group.csv 


### model py-ascet and model parameters

1. The model py-ascet is sir-py-ascet.json
2. The most up to date model parameters are sir_params_v3.json which is the non-normalized parameters, and the sir_parama_concentrations.json are the normalized parameters
3. sir_ic.json is the initial conditions for the un-normalized parameters, while the sir_ic_concentrations.json is the initial conditions for the normalized parameters
