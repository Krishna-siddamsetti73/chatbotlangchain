from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS 
from scraper import retriever
app = Flask(__name__)
CORS(app)
api = Api(app)

class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_query = data.get("query", "")

        # Retrieve relevant information from the vector store
        results = retriever.get_relevant_documents(user_query)

        # Get the most relevant document
        if results:
            response = results[0].page_content
        else:
            response = "Sorry, I couldn't find relevant information."

        return jsonify({"response": response})

# Add the chatbot endpoint
api.add_resource(Chatbot, "/chat")

if __name__ == "__main__":
    app.run(debug=True)
