from preprocess import load_data, get_splits

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

if __name__ == "__main__":
    data, data_info = load_data()
    
    train_ds, val_ds, test_ds = get_splits(data["train"],
        val_size=VAL_SIZE, test_size=TEST_SIZE, shuffle=True)
    print(len(train_ds), len(val_ds), len(test_ds))