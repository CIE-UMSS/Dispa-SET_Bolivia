# Modeling and techno-economic assessment of high variable renewable energy penetration in the Bolivian  power system

Ray Antonio Rojas Candia1, Joseph Adhemar Araoz Ramos1, Sergio Luis Balderrama Subieta (1,2), Jenny Gabriela Peña Balderrama 1, Vicente Senosiain Miquélez 3, Hernan Jaldín Florero 1, and Sylvain Quoilin 2*

1 Energy Research Center, Universidad Mayor de San Simón, Jornad streed, Cochabamba Bolivia

2 Mechanical Engineering Technology TC, Catholic University of LouvainKU Leuven, Geel Campus, Kleinhoefstraat 4 2440 Geel, Belgium

3 Electrical Electronic and Communication Engineering, Public University of Navarre, Campus of Arrosadía – 31006, Pamplona, Spain


Material and models adopted within the study submitted to the International Journal of Sustainable Energy Planning and Management

All material available in the present repository is licensed according to the “European Union Public Licence" EUPL v1.1. It can be redistributed and/or modified under the terms of this license. Citation of the source is requested; please cite the corresponding publication as: R. Rojas et al:  Modeling and techno-economic assessment of high variable renewable energy penetration in the Bolivian  power system, IJSEPM(2019)

### Description
This is input data of the Dispa-SET model, applied to the Bolivian Electrical Interconnected System which has been divided into four zones (North, Central, Oriental and South) and was analized under different VRES (Variable Renewable Energy Sources) penetration levels at 2021. Scenarios are detailed in the following table:

|Year|PV capacity (MW)|PV percentage share|Wind capacity (MW)|Wind percentage share|Total capacity (MW)|
|:--:|:--------------:|:-----------------:|:----------------:|:-------------------:|:-----------------:|
|2016|           0    | 0%                |         27       |  1.5%               | 1855              |
|2021|        174     | 5%                |         165      |  5%                 |    3066           |
|2021|        322     | 10%               |         165      |  5%                 |    3214           |
|2021|        724     | 20%               |         165      |  5%                 |    3615           |
|2021|        964     | 25%               |         165      |  4%                 |    3856           |
|2021|        1240    | 30%               |         165      |  4%                 |    4132           |
|2021|        1929    | 40%               |         165      |  3%                 |    4820           |
|2021|        174     | 5%                |         322      |  10%                |    3223           |
|2021|        174     | 5%                |         725      |  20%                |    3626           |
|2021|        174     | 5%                |         968      |  25%                |    3868           |
|2021|        174     | 4%                |         1244     |  30%                |    4144           |
|2021|        174     | 4%                |         1934     |  40%                |    4835           |
|2021|        234     | 7%                |         165      |  5%                 |    3126           |
|2021|        341     | 10%               |         341      |  10%                |    3409           |
|2021|        584     | 15%               |         584      |  15%                |    3895           |
|2021|        909     | 20%               |         909      |  20%                |    4546           |
|2021|        1364    | 25%               |         1364     |  25%                |    5455           |

### Features
The model is expressed as an optimization problem. Continuous variables include the individual unit dispatched power, the shedded load and the curtailed power generation. The binary variables are the commitment status of each unit. The main model features can be summarized as follows:

- Minimum and maximum power for each unit
- Power plant ramping limits
- Reserves up and down
- Minimum up/down times
- Load Shedding
- Curtailment
- Pumped-hydro storage
- Non-dispatchable units (e.g. wind turbines, run-of-river, etc.)
- Start-up, ramping and no-load costs
- Multi-nodes with capacity constraints on the lines (congestion)
- Constraints on the targets for renewables and/or CO2 emissions
- Yearly schedules for the outages (forced and planned) of each units
- CHP power plants and thermal storage

The demand is assumed to be inelastic to the price signal. The MILP objective function is therefore the total generation cost over the optimization period.

### Quick start

If you want to download the latest version from github for use or development purposes, make sure that you have git and the [anaconda distribution](https://www.anaconda.com/distribution/#download-section)Python 2.7 version installed and type the following:

```bash
git clone https://github.com/energy-modelling-toolkit/Dispa-SET.git
cd Dispa-SET
conda env create  # Automatically creates environment based on environment.yml
source activate dispaset # in Windows: activate dispaset
pip install -e . # Install editable local version
```
The above commands create a dedicated environment so that your anconda configuration remains clean from the required dependencies installed.
To check that everything runs fine, you can build and run a test case by typing:
```bash
dispaset -c ConfigFiles/ConfigTest.xlsx build simulate
```

- Make sure that the path is changed to local Dispa-SET folder in folowing scripts (the procedure is provided in the scripts)
```bash
  - build_and_run.py
  - read_results.py
```
