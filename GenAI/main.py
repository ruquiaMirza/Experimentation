
import os
import google.generativeai as genai

# Configure API key from environment variable

GOOGLE_API_KEY = 'AIzaSyByiW63sAFgRYHCsIUkgwWi_fJNAffS1h4'
if not GOOGLE_API_KEY:
    print("Error: Please set the GOOGLE_API_KEY environment variable.")
    exit()
genai.configure(api_key=GOOGLE_API_KEY)

# #Select the Gemini Pro model
# model = genai.GenerativeModel('gemini-2.0-flash-001')
#
# # Simple text generation
# prompt = "Write a short, imaginative story about a robot who dreams of becoming a gardener."
# response = model.generate_content(prompt)
#
# print(response)


for m in genai.list_models():
    print(m)

#
# print("List of models that support generateContent:\n")
# for m in genai.list_models():
#     for action in m.supported_generation_methods:
#         if action == "predict":
#             print(m.name)