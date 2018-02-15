from flask import Flask, render_template, request, jsonify

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import traceback

from langdetect import detect

import warnings
warnings.filterwarnings("ignore")

import AliasPortal.IOReadWrite as IO

app = Flask(__name__)

#https://www.flashback.org/p63137870#p63137870

@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict")
def predict():
    try:
        # text_field = request.form
        # text1 = text_field['text1']
        # text2 = text_field['text2']

        text1 = request.args.get('text1').strip()
        text2 = request.args.get('text2').strip()

        len_text1 = len(text1)
        len_text2 = len(text2)
        # print(len_text1, len_text2)

        try:
            while (len_text1 > 0 and len_text2 > 0):
                print("inside while")
                continue

                if (len_text1 > 160 and len_text2 > 160):
                    text_lang = detect(text1)

                else:
                    print(len_text1, len_text2)
                    return jsonify(error_msg="Texten är för kort för att bedömmas (minst 160 tecken)")
                    break

            else:
                return jsonify(error_msg="Skriv in en text")

        except Exception:
            # pass
            # return jsonify(error_msg="Skriv in en text")
            return jsonify(error_msg = "Texten är för kort för att bedömmas (minst 160 tecken)")


        # try:
        #     text_lang = detect(text1)
        #
        # except Exception:
        #     return jsonify(error_msg = "Skriv in en text")

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
                # print(df)
                abs_fv = abs(df.diff()).dropna()

                x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]

                same_user_prob, diff_user_prob = IO.return_swe_result(x_test)

            else:
                return jsonify(error_msg = "Språk ej identifierbart")

            return jsonify(
                # pred_class = pred_class,
                same_user_prob = same_user_prob,
                diff_user_prob = diff_user_prob,
                lang = text_lang
            )

        except ValueError:
            traceback.print_exc()
            return jsonify(error_msg = "Ett fel har uppstått")

    except Exception:
        return jsonify(error_msg = traceback.print_exc())


if __name__ == '__main__':
    app.run(debug=True)
