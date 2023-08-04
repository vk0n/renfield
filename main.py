import requests
from transformers import BertTokenizer, BertForQuestionAnswering
import speech_recognition as sr
from espeakng import ESpeakNG

# Ініціалізація BERT-Ukrainian моделі
tokenizer = BertTokenizer.from_pretrained("bertukr-uncased", do_lower_case=True)
model = BertForQuestionAnswering.from_pretrained("bertukr-uncased")

# Ініціалізація eSpeak для синтезу голосу
esng = ESpeakNG()

# Ваш Google API ключ
google_api_key = "ВАШ_КЛЮЧ"

def ask_question(context, question):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    start_positions = torch.tensor([1], dtype=torch.long)  # просто демонстраційне значення
    end_positions = torch.tensor([5], dtype=torch.long)  # просто демонстраційне значення
    outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

def text_to_speech(text):
    esng.voice = 'uk'
    esng.speed = 120
    esng.say(text)

def listen_to_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Скажіть щось...")
        audio = recognizer.listen(source)
    try:
        recognized_text = recognizer.recognize_google(audio, language="uk-UA")
        print("Ви сказали:", recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        print("Не розпізнано")
        return ""
    
def search_with_google(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}"
    response = requests.get(url)
    data = response.json()
    if "items" in data:
        return data["items"][0]["snippet"]
    return "Вибачте, не знайдено інформацію за вашим запитом."

if __name__ == "__main__":
    context = "Київ — столиця та найбільше місто України."
    
    print("Скажіть питання:")
    question = listen_to_voice()
    
    if question:
        answer = ask_question(context, question)
        print(f"Відповідь за BERT: {answer}")
        text_to_speech(answer)
        
        google_answer = search_with_google(question)
        print(f"Відповідь з Google: {google_answer}")
        text_to_speech(google_answer)
    else:
        print("Ви не сказали питання.")


def ask_question(context, question):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    start_positions, end_positions = model(**inputs).start_logits.argmax(), model(**inputs).end_logits.argmax()
    answer = tokenizer.decode(inputs["input_ids"][0][start_positions:end_positions+1])
    return answer


def text_to_speech(text):
    esng.voice = 'uk'
    esng.speed = 120
    esng.say(text)


def listen_to_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Скажіть щось...")
        audio = recognizer.listen(source)
    try:
        recognized_text = recognizer.recognize_google(audio, language="uk-UA")
        print("Ви сказали:", recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        print("Не розпізнано")
        return ""


def search_with_google(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}"
    response = requests.get(url)
    data = response.json()
    if "items" in data:
        return data["items"][0]["snippet"]
    return "Вибачте, не знайдено інформацію за вашим запитом."


def main():
    context = "Київ — столиця та найбільше місто України."

    print("Скажіть питання:")
    question = listen_to_voice()

    if question:
        bert_answer = ask_question(context, question)
        print(f"Відповідь за BERT: {bert_answer}")
        text_to_speech(bert_answer)

        google_answer = search_with_google(question)
        print(f"Відповідь з Google: {google_answer}")
        text_to_speech(google_answer)
    else:
        print("Ви не сказали питання.")

if __name__ == "__main__":
    main()
