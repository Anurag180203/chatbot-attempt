from cdp_chatbot import CDPChatbot

def main():
    chatbot = CDPChatbot()
    
    print("CDP Chatbot: Hello! I can help you with questions about Segment, mParticle, Lytics, and Zeotap.")
    print("CDP Chatbot: Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("CDP Chatbot: Goodbye!")
            break
        
        response = chatbot.answer_question(user_input)
        print("CDP Chatbot:", response)

if __name__ == "__main__":
    main() 