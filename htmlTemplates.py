from bs4 import BeautifulSoup

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''


def sanitize_html(html_string):
    """Safely renders HTML using BeautifulSoup to prevent XSS."""
    soup = BeautifulSoup(html_string, 'lxml')  # Use lxml for parsing
    return soup.prettify()  # Return a prettified string

# Example usage with sanitization (in your main app):
# st.write(sanitize_html(bot_template.replace("{{MSG}}", message.content)), unsafe_allow_html=True)
# st.write(sanitize_html(user_template.replace("{{MSG}}", message.content)), unsafe_allow_html=True)