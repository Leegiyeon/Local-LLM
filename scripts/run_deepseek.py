import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque

# 모델 경로
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# M3 Pro 최적화: MPS(GPU) 지원 여부 확인
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32  # MPS에서는 bfloat16 사용

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map={"": device},
    low_cpu_mem_usage=True
)

# 최근 10개의 대화 히스토리 저장 (더 많은 문맥 유지)
conversation_history = deque(maxlen=10)

# 초기 프롬프트: 최초 1회만 추가
initial_prompt = "AI는 오직 한국어로 대답합니다. 간결하고 정확하게 답변해 주세요."
conversation_history.append(initial_prompt)

def generate_response(input_text):
    # 최근 대화 기록을 포함하여 모델 입력 생성
    history_text = "\n".join(conversation_history) + f"\n사용자: {input_text}\nAI:"
    
    inputs = tokenizer(history_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # MPS 또는 CPU 적용

    with torch.no_grad():  # 메모리 절약
        outputs = model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 대화 기록 저장 (AI: 부분을 제거해 반복 방지)
    conversation_history.append(f"사용자: {input_text}")
    conversation_history.append(response)  # AI 부분을 추가하지 않음

    return response

if __name__ == "__main__":
    print("🔹 DeepSeek 대화 시작 (종료하려면 'exit' 또는 'quit' 입력)")

    while True:
        question = input("\n🟢 사용자: ")

        # 사용자가 'exit' 또는 'quit' 입력 시 종료
        if question.lower() in ["exit", "quit"]:
            print("🔹 대화를 종료합니다.")
            break

        response = generate_response(question)
        print("\n🤖 AI:", response)