from preprocess import load_data, get_splits, preprocess
import tensorflow as tf
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

if __name__ == "__main__":
    data, data_info = load_data()
    
    train_ds, val_ds, test_ds = get_splits(data["train"],
        val_size=VAL_SIZE, test_size=TEST_SIZE, shuffle=True)

    train_ds = preprocess(train_ds)
    val_ds = preprocess(val_ds)
    test_ds = preprocess(test_ds)
    
    for img, lbl in train_ds.take(1):
        print(img.numpy().max())