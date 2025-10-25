progress_html = '''
<div class="loader-container">
  <div class="loader"></div>
  <div class="progress-container">
    <progress value="*number*" max="100"></progress>
  </div>
  <span>*text*</span>
</div>
'''


def make_progress_html(number, text):
    """Creates an HTML progress bar.
    Args:
        number (int): The progress value.
        text (str): The progress text.
    Returns:
        str: The HTML code for the progress bar.
    """
    return progress_html.replace('*number*', str(number)).replace('*text*', text)
