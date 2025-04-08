import joblib
import utils


def test(x_new):
    # loading model
    model, scaler = utils.loading_model()

    # scaling data
    x = utils.transform_using_scaler(scaler, x_new)

    pred = utils.model_prediction(model, x_new)

    pred = utils.prediction_mapping(pred)

    return pred