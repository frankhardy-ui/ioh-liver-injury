\# Intraoperative hypotension and postoperative hepatocellular injury (VitalDB)



Analysis code for the VitalDB cohort study (N = 1427; 195 events).



\## Data

Data are NOT distributed with this code. VitalDB (https://vitaldb.net) is a

public, de-identified perioperative dataset; users must register and sign the

data use agreement themselves.

The locked analysis dataset (one row per case) is read by data\_loader.load()

and is provided as online supplementary material with the manuscript.

The primary outcome is ascertained within 72 hours after surgery.



\## Environment

Python 3.12.  pip install -r requirements.txt



\## Run

step0\_check.py -> step1\_covariates.py -> step2\_table1.py -> step3\_rcs.py ->

figure2\_plot.py -> step4\_m1m5.py -> step5\_aipw.py -> step6\_causal\_forest.py ->

step7\_mediation.py -> step8\_sensitivity.py



\## Acquisition layer

00\_acquire\_vitaldb\_cohort.py --build-cohort --workers 8

00\_acquire\_vitaldb\_cohort.py --compare



\## Licence

MIT



\## Citation

Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity

multi-parameter vital signs database in surgical patients. Sci Data.

2022;9(1):279. doi:10.1038/s41597-022-01411-5

