from modules import data_loader
from modules import sgp4_propagation
from modules import conjunction_detection
from modules import preprocessing
from modules import false_positive
from modules import qae
from modules import reentry_risk
from modules import monte_carlo
from modules import prediction
from modules import classical_security
from modules import qkd

data = data_loader.load_orbital_data()
propagated, failed = sgp4_propagation.propagate_and_save(data)
conjunctions = conjunction_detection.detect_and_save(propagated)

preprocessing.run_and_save()
fp_analysis = false_positive.run_and_save()
qae_comparison = qae.run_and_save()
reentry_analysis = reentry_risk.run_and_save()

mc_results = monte_carlo.run_and_save()
predictions = prediction.run_and_save()

classical_result = classical_security.run_and_save()
qkd_result = qkd.run_and_save()