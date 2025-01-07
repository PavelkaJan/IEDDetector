from src.neural_network.nn_plots import NNPlots
import pickle

with open("diplomka20243012/epileptics_real_and_healthy_real_4s_epochy/all_test_labels_folds.pkl", 'rb') as file:
    all_test_labels_folds = pickle.load(file)

with open("diplomka20243012/epileptics_real_and_healthy_real_4s_epochy/all_test_probs_folds.pkl", 'rb') as file:
    all_test_probs_folds = pickle.load(file)

with open("diplomka20243012/epileptics_real_and_healthy_real_4s_epochy/n_folds.pkl", 'rb') as file:
    cv_splits = pickle.load(file)



NNPlots.plot_mean_roc_curve(
    all_labels=all_test_labels_folds,
    all_probs=all_test_probs_folds,
    n_folds=cv_splits,
    save_path="diplomka20243012/epileptics_real_and_healthy_real_4s_epochy/plots_to_overleaf/rocCurveMeanRealData135mm.pdf"
)
