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
                if(data.same_user_prob <= 85){
                    $('#pred_result').css('color', 'blue');
                    $("#pred_result").text("Olika författare");
                }
                else if(data.same_user_prob > 85){
                    $('#pred_result').css('color', 'green');
                    $("#pred_result").text("Samma författare");
                }
                else if(data.error_msg){
                    $('#pred_result').css('color', 'red');
                    $("#pred_result").text(data.error_msg);
                }
                else{
                    $("#pred_result").text("Vet ej");
                }

                $("#same_user_prob").text("Samma författare: "+ data.same_user_prob + "%")
                $("#diff_user_prob").text("Olika författare: "+ data.diff_user_prob + "%")

                if(data.lang == "sv"){
                    $("#lang").text("Språk: "+ "svenska")
                }
                if(data.lang == "en"){
                    $("#lang").text("Språk: "+ "engelska")
                }

            }
        );
        return false;
    });
});
