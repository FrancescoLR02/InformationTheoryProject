class DatasetManager:
    
    def __init__(self, trainDataset, testDataset, batch_size=100, input_discrete=False, subset_train=None, subset_test=None):
        # ---------------- ORIGINAL DATASETS ----------------
        self.original_train = trainDataset
        self.original_test  = testDataset

        # ---------------- SUBSET SELECTION ----------------
        if subset_train is not None:
            trainData   = trainDataset.data[:subset_train].float() / 255.0
            trainLabels = trainDataset.targets[:subset_train]
        else:
            trainData   = trainDataset.data.float() / 255.0
            trainLabels = trainDataset.targets

        if subset_test is not None:
            testData   = testDataset.data[:subset_test].float() / 255.0
            testLabels = testDataset.targets[:subset_test]
        else:
            testData   = testDataset.data.float() / 255.0
            testLabels = testDataset.targets

        # ---------------- DISCRETIZE (if requested) ----------------
        if input_discrete:
            trainData = (trainData > 0.5).float()
            testData  = (testData  > 0.5).float()

        # ---------------- WRAP INTO TENSORDATASET ----------------
        self.trainDataset = TensorDataset(trainData, trainLabels)
        self.testDataset  = TensorDataset(testData,  testLabels)

        # ---------------- DATALOADERS ----------------
        self.batch_size   = batch_size
        self.train_loader = DataLoader(self.trainDataset, batch_size=batch_size, shuffle=True)
        self.test_loader  = DataLoader(self.testDataset,  batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # SHOW IMAGES
    # -------------------------------------------------------------------------
    def show_images(self, n_rows=5, n_cols=5):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 6))
    
        for ax in axs.flatten():
            img, label = random.choice(self.trainDataset)
            img_np = img.detach().cpu().numpy().squeeze()
            ax.imshow(img_np, cmap='gist_gray')
            ax.set_title(f"Label: {label}")
            ax.set_xticks([])
            ax.set_yticks([])
    
        plt.tight_layout()
        plt.show()


    # -------------------------------------------------------------------------
    # PRINT DATASET INFO
    # -------------------------------------------------------------------------
    def get_info_dataset(self):
        print(f"Length of train dataset: {len(self.trainDataset)}")
        print(f"Length of test dataset: {len(self.testDataset)}\n")
        print(f"There are {self.batch_size} images per batch (last batch may differ)\n")

        batch_data, batch_labels = next(iter(self.train_loader))
        print(f"(N° of train batches: {len(self.train_loader)})")
        print("TRAIN BATCH SHAPE")
        print(f"\tData: {batch_data.shape}")
        print(f"\tLabels: {batch_labels.shape}\n")

        batch_data, batch_labels = next(iter(self.test_loader))
        print(f"(N° of test batches: {len(self.test_loader)})")
        print("TEST BATCH SHAPE")
        print(f"\tData: {batch_data.shape}")
        print(f"\tLabels: {batch_labels.shape}\n")