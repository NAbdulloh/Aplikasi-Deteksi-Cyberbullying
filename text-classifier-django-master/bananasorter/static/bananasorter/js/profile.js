var $class_api = '/api/classifiers/';
var $user_api = '/api/users/';
var $form = $('.classifier_form')
var $csrf = $form.find('input[name="csrfmiddlewaretoken"]')
var $name = $form.find('input[name="name"]')
var $user = $form.find('input[name="user_id"]')


function append_classifier( classifier ) {
    $('.classifier_list').append(
        $('<li>').append(
            $("<a href='/" + classifier.id + "'>").text(classifier.name)));
};


function get_classifiers() {
    $.get($class_api + '?user_id=' + $user.val(), function( data ) {
    data.results.forEach(function( classifier ) {
        append_classifier( classifier );
    });
})};
get_classifiers()


$form.submit( function( event ) {
    event.preventDefault();

    var $data = {
        csrfmiddlewaretoken: $csrf.val(),
        name: $name.val(),
        owner: $user_api + $user.val() + '/',
    };

    $.ajax({
        url: $class_api,
        method: 'POST',
        data: $data,
        success: function() {
            console.log('I think it worked!');
            $.get($class_api, function( data ) {
                append_classifier(data.results[0])
            });
        }
    });
});
