var $class_api = '/api/classifiers/';
var $user_api = '/api/users/';
var $cat_api = '/api/categories/';
var $use_form = $('#classify')
var $cat_form = $('.category_form')
var $del_form = $('#delete')
var $classifier = $use_form.find('input[name="classifier_id"]')
var $csrf = $cat_form.find('input[name="csrfmiddlewaretoken"]')
var $name = $cat_form.find('input[name="name"]')
var $training_data = $cat_form.find('textarea[name="training_data"]')
// var $user = $form.find('input[name="user_id"]')


function append_category( category ) {
    $('.category_list').append(
        $('<li>').text(category.name));
};


function get_classifiers() {
    $.get(
        $cat_api + '?classifier_id=' + $classifier.val(),
        function( data ) {
            data.results.forEach(function( category ) {
                append_category( category );
            })
    });
};
get_classifiers()

$cat_form.submit( function( event ) {
    event.preventDefault();

    var $data = {
        csrfmiddlewaretoken: $csrf.val(),
        name: $name.val(),
        training_data: $training_data.val(),
        classifier: $class_api + $classifier.val() + '/',
    }

    $.ajax({
        url: $cat_api,
        method: 'POST',
        data: $data,
        success: function() {
            console.log('I think it worked!');
            append_category($data);
        }
    });
});
