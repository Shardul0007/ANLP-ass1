import random
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_sizes,
        boundaries,
        shuffle=True,
        seed=42,
    ):
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self.boundaries = boundaries
        self.shuffle = shuffle
        self.seed = seed

        self.lengths = [
            len(dataset.cipher_lines[i])
            for i in range(len(dataset))
        ]

        self.buckets = self._create_buckets()

    def _bucket_index(self, length):
        for i, boundary in enumerate(self.boundaries):
            if length <= boundary:
                return i

        return len(self.boundaries)

    def _create_buckets(self):
        buckets = [
            []
            for _ in range(len(self.boundaries) + 1)
        ]

        for index, length in enumerate(self.lengths):
            bucket = self._bucket_index(length)
            buckets[bucket].append(index)

        return buckets

    def __iter__(self):
        rng = random.Random(self.seed)

        all_batches = []

        for bucket_index, indices in enumerate(self.buckets):
            indices = indices.copy()

            if self.shuffle:
                rng.shuffle(indices)

            batch_size = self.batch_sizes[bucket_index]

            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]

                if batch:
                    all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        total = 0

        for bucket_index, indices in enumerate(self.buckets):
            batch_size = self.batch_sizes[bucket_index]

            total += (
                len(indices) + batch_size - 1
            ) // batch_size

        return total