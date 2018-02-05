from flask import Flask, render_template, request, jsonify

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import traceback

from langdetect import detect

import warnings
warnings.filterwarnings("ignore")

import AliasPortal.IOReadWrite as IO

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict")
def predict():
    try:
        text1 = request.args.get('text1')
        text2 = request.args.get('text2')

        try:
            text_lang = detect(text1)

        except Exception:
            return jsonify(error_msg = "Please insert text")

        try:
            if text_lang == "en":
                fv_dataframe = IO.create_english_feature_vector(text1, text2)

                df = pd.DataFrame(fv_dataframe)
                abs_fv = abs(df.diff()).dropna()

                x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]

                same_user_prob, diff_user_prob = IO.return_eng_result(x_test)

            elif text_lang == "sv":
                fv_dataframe = IO.create_swedish_feature_vector(text1, text2)

                df = pd.DataFrame(fv_dataframe)
                abs_fv = abs(df.diff()).dropna()

                x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]

                same_user_prob, diff_user_prob = IO.return_swe_result(x_test)

            else:
                return jsonify(error_msg = "Language problem")

            return jsonify(
                # pred_class = pred_class,
                same_user_prob = same_user_prob,
                diff_user_prob = diff_user_prob,
                lang = text_lang
            )

        except ValueError:
            traceback.print_exc()
            return jsonify(error_msg = "Something is wrong !!!")

    except Exception:
        return jsonify(error_msg = traceback.print_exc())


if __name__ == '__main__':
    app.run(debug=True)
