from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os

# Create your views here.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn tuyệt đối đến thư mục chứa mô hình
model_dir = os.path.join(BASE_DIR, 'models', 'bart_summarization_model')

# Tải mô hình và tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
def summarize(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

    # Decode the summary tokens back into text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def index(request):
    if request.method == 'POST':
        text = request.POST.get('text')

        if text:
            summarized_text = summarize(text)

            return render(request, 'home/summary_result.html', {
                'original_text': text,
                'summarized_text': summarized_text,
            })
        else:
            return HttpResponse("No text provided", status=400)

        # Nếu là GET, trả về form để người dùng nhập văn bản
    return render(request, 'home/index.html')