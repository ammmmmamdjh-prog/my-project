# تمرین ۱ - Fine-tuning مدل‌های زبانی برای طبقه‌بندی متن فارسی
# Dataset: ParsiAI/snappfood-sentiment-analysis
# Models: ParsBERT و XLM-RoBERTa
#
# اجرا در Google Colab (رایگان + GPU):
#   1) Runtime > Change runtime type > GPU
#   2) کل این فایل را در یک سلول اجرا کنید.
#
# اجرا روی سرور معمولی (بدون Jupyter):
#   python3 Exercise_1_Text_Classification_fixed.py
#
# ============================================================
# فهرست اصلاحات نسبت به نسخه اولیه:
#   [FIX 1] سینتکس Colab «!pip install» باعث SyntaxError روی هر سرور
#           معمولی می‌شود؛ با روش سازگار با Colab و سرور جایگزین شد.
#   [FIX 2] خطای «invalid literal for int() with base 10: 'SAD'»:
#           ستون label در این دیتاست رشته‌ای است (SAD/HAPPY) و ستون
#           عددی label_id است. تبدیل لیبل حالا از label_id انجام
#           می‌شود و نگاشت id2label/label2id از جفت‌های واقعی
#           (label_id, label) خودِ دیتاست ساخته می‌شود.
#   [FIX 3] تابع display() فقط در Jupyter/Colab وجود دارد و روی سرور
#           NameError می‌دهد؛ با print جایگزین شد.
#   [FIX 4] چون save_strategy="no" بود، هیچ مدلی ذخیره نمی‌شد و بخش
#           پایانی pipeline همیشه با خطا skip می‌شد. حالا مدل و
#           توکنایزر با trainer.save_model ذخیره می‌شوند.
#   [FIX 5] ستون‌های غیرورودی (comment و label و label_id) بعد از
#           توکنایز کامل حذف می‌شوند؛ در بسیاری از نسخه‌های
#           transformers، باقی‌ماندن ستون رشته‌ای comment باعث خطای
#           DataCollatorWithPadding هنگام ساخت batch می‌شود.
#   [FIX 6] device=-1 در pipeline منسوخ شده است؛ از "cpu" استفاده شد.
#   [FIX 7] اندازه داده/تعداد epoch/طول توکن با متغیر محیطی قابل
#           تنظیم است (پیش‌فرض‌ها همان مقادیر Colab: 3000/500/500
#           و 3 epoch). روی سرور بدون GPU می‌توان کمترشان کرد:
#     مثال: TRAIN_SIZE=400 VALID_SIZE=100 TEST_SIZE=200 EPOCHS=1 \
#           MAX_LENGTH=96 python3 Exercise_1_Text_Classification_fixed.py
#   [FIX 8] با OPTIM=adafactor می‌توان روی سرورهای کم‌حافظه (RAM < 8G)
#           از OOM شدن مدل‌های بزرگ مثل XLM-RoBERTa جلوگیری کرد.
#           پیش‌فرض همان adamw_torch است.
#   [FIX 9] BATCH_SIZE و EVAL_BATCH_SIZE هم با متغیر محیطی قابل تنظیم
#           شدند و pin_memory روی CPU غیرفعال می‌شود.
#   [FIX 10] با MODELS=parsbert یا MODELS=xlmr می‌توان هر مدل را جدا
#           آموزش داد (روی سرورهای ضعیف مفید است). نتایج هر اجرا در
#           exercise1_results.jsonl ذخیره و در جدول نهایی ادغام می‌شود.
#           پیش‌فرض: MODELS=parsbert,xlmr (هر دو، مثل نسخه اصلی)
#   [FIX 11] با GRAD_CKPT=1 حافظه آموزش با gradient checkpointing
#           کم می‌شود (برای RAM کم، مثلاً مدل XLM-RoBERTa با RAM 4G).
#   [FIX 12] low_cpu_mem_usage همیشه فعال شد و با MODEL_DTYPE=bf16
#           می‌توان حافظه پارامتر/گرادیان را هم نصف کرد.
#   [FIX 13] با THREADS می‌توان تعداد threadهای CPU تورچ را محدود کرد
#           (کاهش مصرف حافظه) و با DEBUG_MEM=1 لاگ حافظه چاپ می‌شود.
#   [FIX 14] با FREEZE_EMB=1 جدول embedding فریز می‌شود تا گرادیان بزرگ
#           آن (در XLM-R حدود 400MB) ساخته نشود؛ برای RAM کم مفید است.
# ============================================================

# ============================================================
# 1. نصب کتابخانه‌ها
# ============================================================
# [FIX 1] به‌جای «!pip» که فقط در Jupyter/Colab کار می‌کند، از
# subprocess استفاده می‌کنیم تا هم در Colab و هم روی سرور معمولی کار کند.

import subprocess
import sys

def _install_packages():
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install", "-q", "-U",
            "transformers", "datasets", "accelerate", "evaluate",
            "scikit-learn", "sentencepiece",
        ]
    )

try:
    import transformers  # noqa: F401
    import datasets  # noqa: F401
    import evaluate  # noqa: F401
    import sklearn  # noqa: F401
    import sentencepiece  # noqa: F401
    print("All required packages are already installed.")
except ImportError:
    print("Installing required packages ...")
    _install_packages()

# ============================================================
# 2. Import
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

set_seed(42)

print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# تنظیمات حجم داده و آموزش
# ============================================================
# [FIX 7] پیش‌فرض‌ها دقیقاً همان مقادیر نسخه اصلی برای Colab GPU است.
# برای اجرای سبک‌تر (مثلاً روی CPU سرور) از متغیر محیطی استفاده کنید.

TRAIN_SIZE = int(os.environ.get("TRAIN_SIZE", 3000))
VALID_SIZE = int(os.environ.get("VALID_SIZE", 500))
TEST_SIZE = int(os.environ.get("TEST_SIZE", 500))
NUM_EPOCHS = int(os.environ.get("EPOCHS", 3))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 128))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", 50))
OPTIM = os.environ.get("OPTIM", "adamw_torch")  # [FIX 8]
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))       # پیش‌فرض نسخه اصلی
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 16))  # پیش‌فرض نسخه اصلی
MODELS = os.environ.get("MODELS", "parsbert,xlmr")  # [FIX 10]
SELECTED_MODELS = [m.strip().lower() for m in MODELS.split(",") if m.strip()]
RESULTS_JSONL = "exercise1_results.jsonl"
GRAD_CKPT = os.environ.get("GRAD_CKPT", "0") == "1"  # [FIX 11] صرفه‌جویی حافظه
MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "fp32")  # [FIX 12] fp32 یا bf16
THREADS = int(os.environ.get("THREADS", "0"))  # [FIX 13] محدودکردن threadهای CPU
DEBUG_MEM = os.environ.get("DEBUG_MEM", "0") == "1"  # لاگ حافظه برای عیب‌یابی
FREEZE_EMB = os.environ.get("FREEZE_EMB", "0") == "1"  # [FIX 14] فریز embedding

def _dbg(label):
    if DEBUG_MEM:
        try:
            rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1024
            print("[MEM] %s: rss=%.0fMB" % (label, rss), flush=True)
        except Exception:
            pass

if THREADS > 0:
    torch.set_num_threads(THREADS)  # [FIX 13]

_dbg("after imports")

# ============================================================
# 3. بارگذاری دیتاست Snappfood
# ============================================================

dataset = load_dataset("ParsiAI/snappfood-sentiment-analysis")

_dbg("after dataset load")

print(dataset)
print("\nFeatures:")
print(dataset["train"].features)

print("\nFirst example:")
print(dataset["train"][0])

# ============================================================
# 4. تشخیص خودکار label
# ============================================================
# [FIX 2] ساختار واقعی دیتاست:
#   label    -> رشته‌ای است: 'HAPPY' یا 'SAD'
#   label_id -> عددی است:  0 یا 1
# پس نگاشت را از جفت‌های واقعی (label_id, label) خود دیتاست می‌سازیم.

pairs = set(zip(dataset["train"]["label_id"], dataset["train"]["label"]))
id2label = {int(k): str(v) for k, v in pairs}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

print("\nid2label:", id2label)
print("label2id:", label2id)
print("Number of labels:", num_labels)

# ============================================================
# 5. انتخاب اندازه دیتاست
# ============================================================

train_data = dataset["train"].shuffle(seed=42).select(
    range(min(TRAIN_SIZE, len(dataset["train"])))
)

valid_data = dataset["validation"].shuffle(seed=42).select(
    range(min(VALID_SIZE, len(dataset["validation"])))
)

test_data = dataset["test"].shuffle(seed=42).select(
    range(min(TEST_SIZE, len(dataset["test"])))
)

print("\nSelected data:")
print("Train:", len(train_data))
print("Validation:", len(valid_data))
print("Test:", len(test_data))

# ============================================================
# 6. معیارهای ارزیابی
# ============================================================

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, true_labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=true_labels
    )

    f1 = f1_metric.compute(
        predictions=predictions,
        references=true_labels,
        average="weighted"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

# ============================================================
# 7. تابع Fine-tuning
# ============================================================

def train_text_model(model_name, model_display_name, output_dir):

    print("\n" + "=" * 80)
    print("Starting:", model_display_name)
    print("Model:", model_name)
    print("=" * 80)

    start_time = time.time()

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch["comment"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_tokenized = train_data.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing train"
    )

    valid_tokenized = valid_data.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing validation"
    )

    test_tokenized = test_data.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing test"
    )

    # -----------------------------
    # Label
    # -----------------------------
    # [FIX 2] ستون عددی دیتاست label_id است (نه label).

    def convert_labels(example):
        example["labels"] = int(example["label_id"])
        return example

    train_tokenized = train_tokenized.map(
        convert_labels,
        desc="Preparing train labels"
    )

    valid_tokenized = valid_tokenized.map(
        convert_labels,
        desc="Preparing validation labels"
    )

    test_tokenized = test_tokenized.map(
        convert_labels,
        desc="Preparing test labels"
    )

    # -----------------------------
    # حذف ستون‌های اضافه
    # -----------------------------
    # [FIX 5] فقط ستون‌های ورودی مدل (input_ids, attention_mask,
    # token_type_ids و ...) به‌علاوه labels باقی می‌مانند؛ ستون‌های
    # comment و label و label_id حذف می‌شوند تا DataCollator خطا ندهد.

    def keep_only_model_inputs(ds):
        keep = set(tokenizer.model_input_names) | {"labels"}
        drop = [c for c in ds.column_names if c not in keep]
        return ds.remove_columns(drop)

    train_tokenized = keep_only_model_inputs(train_tokenized)
    valid_tokenized = keep_only_model_inputs(valid_tokenized)
    test_tokenized = keep_only_model_inputs(test_tokenized)

    print("\nTrain columns:", train_tokenized.column_names)
    _dbg("after tokenization")

    # -----------------------------
    # Model
    # -----------------------------
    # [FIX 12] low_cpu_mem_usage از ساخته‌شدن کپی کامل state_dict در RAM
    # جلوگیری می‌کند (اوج حافظه هنگام load تقریباً نصف می‌شود). با
    # MODEL_DTYPE=bf16 هم پارامترها و گرادیان‌ها نصف می‌شوند
    # (برای مدل‌های بزرگ مثل XLM-R روی سرورهای کم‌حافظه).
    model_dtype = torch.bfloat16 if MODEL_DTYPE == "bf16" else None

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        low_cpu_mem_usage=True,
        dtype=model_dtype
    )

    _dbg("after model load (%s)" % model_display_name)

    # [FIX 14] در حالت کم‌حافظه می‌توان جدول embedding را فریز کرد؛
    # گرادیان آن (برای XLM-R حدود 400 مگابایت) دیگر ساخته نمی‌شود.
    # روی کیفیت طبقه‌بندی معمولاً تأثیر زیادی ندارد.
    if FREEZE_EMB:
        frozen = 0
        for name, param in model.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
                frozen += param.numel()
        print("Frozen embedding parameters: %.1fM" % (frozen / 1e6))

    # -----------------------------
    # Data collator
    # -----------------------------
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,

        eval_strategy="epoch",
        save_strategy="no",

        learning_rate=2e-5,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        weight_decay=0.01,

        logging_steps=LOGGING_STEPS,

        report_to="none",

        fp16=torch.cuda.is_available(),

        # [FIX 9] روی ماشین‌های بدون GPU، pin_memory فقط هشدار و مصرف
        # حافظه اضافه می‌دهد؛ فقط با GPU فعال می‌شود.
        dataloader_pin_memory=torch.cuda.is_available(),

        # [FIX 11] برای مدل‌های بزرگ روی سرورهای کم‌حافظه: با محاسبه مجدد
        # فعال‌سازی‌ها، مصرف RAM به‌شدت کم می‌شود (کمی کندتر).
        gradient_checkpointing=GRAD_CKPT,

        optim=OPTIM,  # [FIX 8]

        seed=42
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    _dbg("after trainer ready")

    # -----------------------------
    # Training
    # -----------------------------
    trainer.train()

    _dbg("after training")

    # -----------------------------
    # [FIX 4] ذخیره مدل و توکنایزر برای استفاده در بخش pipeline
    # -----------------------------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("\nModel saved to:", output_dir)

    # -----------------------------
    # Final test evaluation
    # -----------------------------
    print("\nEvaluating on TEST set...")

    test_results = trainer.evaluate(
        eval_dataset=test_tokenized
    )

    elapsed = time.time() - start_time
    training_minutes = elapsed / 60

    result = {
        "Dataset": "Snappfood",
        "Model": model_display_name,
        "Model_ID": model_name,
        "Accuracy": test_results.get("eval_accuracy"),
        "F1": test_results.get("eval_f1"),
        "Training_Time_Minutes": round(training_minutes, 2),
        "Epochs": NUM_EPOCHS,
        "Train_Samples": len(train_data),
        "Validation_Samples": len(valid_data),
        "Test_Samples": len(test_data)
    }

    print("\n" + "-" * 80)
    print("FINAL TEST RESULT")
    print("-" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    return result

# ============================================================
# 8. آموزش ParsBERT
# ============================================================

parsbert_result = None
xlmr_result = None

if "parsbert" in SELECTED_MODELS:
    parsbert_result = train_text_model(
        model_name="HooshvareLab/bert-fa-base-uncased",
        model_display_name="ParsBERT",
        output_dir="./results-parsbert"
    )
else:
    print("\nSkipping ParsBERT (MODELS=%s)" % MODELS)

# ============================================================
# 9. آزاد کردن حافظه GPU
# ============================================================

import gc

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
# 10. آموزش XLM-RoBERTa
# ============================================================

if "xlmr" in SELECTED_MODELS:
    xlmr_result = train_text_model(
        model_name="xlm-roberta-base",
        model_display_name="XLM-RoBERTa",
        output_dir="./results-xlmr"
    )
else:
    print("\nSkipping XLM-RoBERTa (MODELS=%s)" % MODELS)

# ============================================================
# 11. جدول نهایی نتایج
# ============================================================

# [FIX 10] اگر بخشی از مدل‌ها در اجراهای قبلی آموزش دیده باشند،
# نتایج آن‌ها از exercise1_results.jsonl خوانده و ادغام می‌شود.

import json

def load_previous_results(path):
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows

def append_results(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

current_results = [r for r in [parsbert_result, xlmr_result] if r is not None]

if current_results:
    append_results(RESULTS_JSONL, current_results)

merged = {}
# اول نتایج قبلی، بعد نتایج فعلی (نتایج فعلی اولویت دارند)
for row in load_previous_results(RESULTS_JSONL) + current_results:
    merged[row["Model_ID"]] = row

results_table = pd.DataFrame(list(merged.values()))

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

# [FIX 3] display() فقط در Jupyter تعریف شده است؛ روی سرور NameError می‌دهد.
print(results_table.to_string(index=False))

# ذخیره جدول برای گزارش
results_table.to_csv(
    "exercise1_results.csv",
    index=False
)

print("\nResults saved to: exercise1_results.csv")

# ============================================================
# 12. نتیجه بهتر
# ============================================================

best_accuracy_row = results_table.loc[
    results_table["Accuracy"].idxmax()
]

best_f1_row = results_table.loc[
    results_table["F1"].idxmax()
]

print("\nBest model by Accuracy:")
print(best_accuracy_row["Model"])
print("Accuracy:", best_accuracy_row["Accuracy"])

print("\nBest model by F1:")
print(best_f1_row["Model"])
print("F1:", best_f1_row["F1"])

# ============================================================
# 13. تست یک جمله فارسی با مدل ParsBERT
# ============================================================
# [FIX 4] حالا چون مدل ذخیره شده است، این بخش واقعاً کار می‌کند.
# [FIX 6] device=-1 منسوخ شده است؛ از "cpu" استفاده می‌کنیم.

from transformers import pipeline

parsbert_dir = "./results-parsbert"
parsbert_saved = os.path.exists(os.path.join(parsbert_dir, "config.json"))

if not parsbert_saved:
    print("\nOptional prediction section skipped:")
    print("ParsBERT model folder not found (%s)." % parsbert_dir)
else:
    try:
        classifier = pipeline(
            "text-classification",
            model="./results-parsbert",
            tokenizer="./results-parsbert",
            device=0 if torch.cuda.is_available() else "cpu"
        )

        test_texts = [
            "غذا خیلی خوشمزه بود و کیفیت عالی داشت.",
            "غذا سرد بود و اصلاً راضی نبودم."
        ]

        print("\nSample predictions:")

        for text in test_texts:
            prediction = classifier(text)
            print("\nText:", text)
            print("Prediction:", prediction)

    except Exception as e:
        print("\nOptional prediction section skipped:")
        print(e)

# ============================================================
# پایان تمرین ۱
# ============================================================
print("\nExercise 1 finished.")
print("File: exercise1_results.csv")
