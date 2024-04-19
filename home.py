from flask import Flask, render_template

# Create a Flask application
app = Flask(__name__)

# Define a route to render the HTML file
@app.route('/')
def index():
    # Render the HTML file named 'index.html' located in the 'templates' folder
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
