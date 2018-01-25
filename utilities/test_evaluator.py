from sklearn.metrics import confusion_matrix, classification_report


def evaluator(filename, y_actual_validate, y_pred_validate, y_actual_test, y_pred_test, duration):
    conf_matrix_validate = confusion_matrix(y_actual_validate, y_pred_validate)
    class_report_validate = classification_report(y_actual_validate, y_pred_validate)
    accuracy_validate = sum([row[i] for i, row in enumerate(conf_matrix_validate)])/sum(sum(conf_matrix_validate))

    conf_matrix_test = confusion_matrix(y_actual_test, y_pred_test)
    class_report_test = classification_report(y_actual_test, y_pred_test)
    accuracy_test = sum([row[i] for i, row in enumerate(conf_matrix_test)]) / sum(sum(conf_matrix_test))

    with open(filename, 'w') as f:
        f.write("****** VALIDATE DATASET ******\n")
        f.write("Accuracy: " + str(accuracy_validate * 100) + '%\n\n')
        f.write(str(conf_matrix_validate) + '\n\n')
        f.write(class_report_validate + '\n\n')
        f.write("****** TEST DATASET ******\n")
        f.write("Accuracy: " + str(accuracy_test * 100) + '%\n\n')
        f.write(str(conf_matrix_test) + '\n\n')
        f.write(class_report_test + '\n\n')
        f.write("Time taken: " + str(duration))
