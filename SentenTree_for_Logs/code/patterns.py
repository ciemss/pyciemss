import collections
import pandas as pd


class Encoder(collections.UserDict):
    """Encodes/decodes strings to more symbolic representation."""

    def __init__(self, encode_to="number"):
        """
        encode_to -- 'number' -- Will encode values to integers
                     'character' -- Will encode values to single characters (can fail due to running out of characters)
        """
        super().__init__(self)
        self.i = 0
        self.options = None
        if encode_to == "character":
            self.options = (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
            )
        self._decoder = None

    def get(self, key):
        if key not in self:
            self.__missing__(key)
        return super().get(key)

    def __missing__(self, key):
        if self.options is not None:
            self[key] = self.options[self.i]
        else:
            self[key] = self.i
        self.i += 1
        self._decoder = None  # Invalidate decoder after we add something

    def decode_all(self, patterns):
        """Decode all matches in the patterns passed.
        This is a helper for workign with the results of the prefixspan library.
        patterns --  Iterable of pairs (size, pattern).
        returns -- list of pairs (size, decoded-pattern)
        """
        return [(count, self.decode(seq=seq)) for count, seq in patterns]

    def encode(self, value=None, seq=None):
        if seq is not None:
            return [self[v] for v in seq]
        else:
            return self[value]

    def decode(self, value=None, seq=None):
        """Decode a specific value.
        If seq is passed, decodes a SEQUENCE of values.
        Otherwise, decodes a single value.
        """
        decoder = self.decoder()
        if seq is not None:
            return [decoder[v] for v in seq]
        else:
            return decoder[value]

    def decoder(self):
        """Get a decoder dictionary.  Will not reflect future updates to the encoder"""
        if not self._decoder:
            self._decoder = dict(zip(self.values(), self.keys()))
        return self._decoder


def encode(sequence, field=None, encoder=None):
    """Encode a sequence of arbitrary entries as a sequence of numbers.

    If used as part of an apply, the encoder should be provided.  Otherwise
    different sequences will each have their own encoder.

    seqeunce -- dataframe or list.
    field -- If sequence is a dataframe, which column to use (default is first column)
    encoder -- Encoder to use (a new one is created by default)
    returns -- Encoded sequence as a list, encoder
    """

    class Encoded(collections.abc.Sequence):
        def __init__(self, seq, encoder):
            self.encoder = encoder
            self.seq = seq

        def encoder(self):
            """Encoder used to create the this sequence."""
            return self.encoder

        def decoded(self):
            """Return a decoded varaint.  ACTUALLY DECODES so might not exactly equal the input.
            Assumes the decoder has a 'decode(seq=X)' variant.
            """
            return encoder.decode(seq=self)

        def __getitem__(self, i):
            return self.seq[i]

        def __len__(self):
            return len(self.seq)

        def __repr__(self):
            return str(self.seq) + "+encoder"

    if isinstance(sequence, pd.DataFrame):
        if field is None:
            field = sequence.columns[0]
        sequence = sequence[field].values

    if encoder is None:
        encoder = Encoder()
    return Encoded([encoder.get(e) for e in sequence], encoder)


def to_maximal(patterns):
    """Filters out any pattern that is a subset of any other pattern
    (This may be slow for thousands of patterns, but its just fine for hundreds of them.)
    """

    def in_any(focus, others):
        for other in others:
            if focus in other and focus != other:
                return True
        return False

    support, patterns = zip(*patterns)
    string_pats = [" ".join(str(e) for e in pat) for pat in patterns]
    return [
        (support, pattern)
        for support, pattern, string_pat in zip(support, patterns, string_pats)
        if not in_any(string_pat, string_pats)
    ]


def get_exemplars(hdbscan, sequences, n=1, *, label=None):
    """
    Given an (trained) hdbscan classifier, get n examples from each identified cluster.
    If there are not n exemplars, returns as many as it can.
    hdbscan -- hdbscan object trained on a set of documents
    sequences -- Dataframe where the first index level's k-th item is the id for
                the k-th vector the hdbscan object was trained on
    n -- Number of exemplars to return

    returns Dataframe with document ID on the index, cluster ID and sequence in the columns
    """
    if not label:
        label = sequences.columns[0]

    clusters = pd.DataFrame(
        {"outlier_score": hdbscan.outlier_scores_, "cluster_id": hdbscan.labels_},
        index=sequences.index.unique(0),
    )

    def top_n(grp):
        name = grp.name
        grp = grp.sort_values("outlier_score")
        ids = grp.index[:n]
        seqs = [sequences.loc[id][label].values for id in ids]
        return pd.DataFrame({"sequence": seqs}, index=ids)

    return (
        clusters[clusters["cluster_id"] != -1]
        .groupby("cluster_id")
        .apply(top_n)
        .swaplevel(0, 1)
    )
