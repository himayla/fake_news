from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

id2label = {0: "FAKE", 1: "REAL"}
label2id = {"FAKE": 0, "REAL": 1}

print("Making predictions...")

model = AutoModelForSequenceClassification.from_pretrained("bert/checkpoint-3")

fake = "Michelle Obama Deletes Hillary Clinton From Twitter"
fake1 = "Pope Francis Shocks World, Endorses Donald Trump For President"
fake2 = "Health care reform legislation is likely to mandate free sex change surgeries."
real = "Russia Targets Ukraine With Overnight Drone Strikes"

input_sentence = fake

encoding = tokenizer(input_sentence, return_tensors="pt")
print(f"Input sentence: {input_sentence}")

outputs = model(**encoding)

logits = outputs.logits

predicted_class_id = logits.argmax().item()
prediction = model.config.id2label[predicted_class_id]
print(prediction)