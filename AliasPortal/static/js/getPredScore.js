/**
 * Created by amendrashrestha on 18-02-02.
 */

$(document).ready(function() {

    $(document).ajaxStart(function () {
        $("#loading").show();
    }).ajaxStop(function () {
        $("#loading").hide();
        $('.sidebar-middle').show();
    });

    $('button#calculate').bind('click', function() {
        $.getJSON('/predict', {
                text1: $('textarea[name="text1"]').val(),
                text2: $('textarea[name="text2"]').val()
            }, function(data) {
                console.log(data)
//                if (data.pred_class == 1){
//                    $("#pred_result").text("Diff User");
//                }
                if(data.same_user_prob < 0.7){
                    $("#pred_result").text("Less than 70%");
                }
                else if(data.same_user_prob > 0.7){
                    $("#pred_result").text("Same User");
                }
                else if(data.error_msg){
                    $("#pred_result").text(data.error_msg);
                }
                else{
                    $("#pred_result").text("Diff User");
                }
                $("#same_user_prob").text("Same User Prob: "+ data.same_user_prob)
                $("#diff_user_prob").text("Diff User Prob: "+ data.diff_user_prob)

                if(data.lang == "sv"){
                    $("#lang").text("Language: "+ "Swedish")
                }
                if(data.lang == "en"){
                    $("#lang").text("Language: "+ "English")
                }

            }
        );
        return false;
    });
});
