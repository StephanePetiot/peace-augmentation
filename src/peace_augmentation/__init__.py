import random
from typing import List

import nltk
from flair.data import Sentence

from .easy_data_augmentation import (
    get_only_chars,
    random_deletion,
    random_insertion,
    random_swap,
    synonym_replacement,
)
from .replace_named_entities import find_closest_embeddings, get_word_vector
from .utils import (  # add_exp_to_sent,; vectors_list,
    fasttext_doc_emb,
    get_synonyms,
    replace_exp_to_sent,
    scalarity_list,
    special_tokens,
    speculative_list,
    tagger_ner,
    tagger_pos,
)


def check_and_download_nltk_data():
    required_resources = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']

    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)


check_and_download_nltk_data()


class Augmentation:
    def __init__(
        self,
        text,
    ):
        self.text = text

    def apply(self, method, **kwargs):
        return eval(f"self.{method}")(self.text, **kwargs)

    def easy_data_augmentation(
        self,
        sent: str,
        alpha_sr: int = 0.1,
        alpha_ri: int = 0.1,
        alpha_rs: int = 0.1,
        p_rd: int = 0.1,
        num_aug: int = 2,
        add_original: bool = False,
    ):
        """
        Data augmentation with method `eda`. Given an input sentence `sent`, i) it
        randomly replaces a non-stopword expression with a synonym using Word-net;
        ii) inserts a synonym of a non-stopword word in a random position; iii)
        chooses two words of the sentence and swap their positions; iv) removes each
        word in the sentence with a certain probability. Only one of the four
        operations at a time is applied to a sentence.

        parameters:
            - `sent` (str) sentence to augment.
            - `alpha_sr` (int) percent of the words in a sentence to be changed in
            synonym replacement.
            - `alpha_ri` (int) percent of the words in a sentence to be changed in
            random insertion.
            - `alpha_rs` (int) percent of the words in a sentence to be changed in
            random swap.
            - `p_rd` (int) percent of the words in a sentence to be changed in
            random deletion.
            - `num_aug` (int) number of augmented sentences per technique.
            - `add_original` (bool) add original sentence to the output.

        returns:
            - augmented_sentences (List[str]) list of augmented sentences.
        """

        only_char_sentence = get_only_chars(sent)
        words = only_char_sentence.split(" ")
        words = [word for word in words if word != ""]
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1

        # synonym replacement
        if alpha_sr > 0:
            n_sr = max(1, int(alpha_sr * num_words))
            for _ in range(num_new_per_technique):
                a_words = synonym_replacement(words, n_sr)
                augmented_sentences.append(" ".join(a_words))

        # random insertion
        if alpha_ri > 0:
            n_ri = max(1, int(alpha_ri * num_words))
            for _ in range(num_new_per_technique):
                a_words = random_insertion(words, n_ri)
                augmented_sentences.append(" ".join(a_words))

        # random swap
        if alpha_rs > 0:
            n_rs = max(1, int(alpha_rs * num_words))
            for _ in range(num_new_per_technique):
                a_words = random_swap(words, n_rs)
                augmented_sentences.append(" ".join(a_words))

        # random deletion
        if p_rd > 0:
            for _ in range(num_new_per_technique):
                a_words = random_deletion(words, p_rd)
                augmented_sentences.append(" ".join(a_words))

        augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
        random.shuffle(augmented_sentences)

        # Trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        if add_original:
            # append the original sentence
            augmented_sentences.append(sent)

        return augmented_sentences

    def replace_adjectives(
        self, sent: str, pos: str = "a", only_hyponyms: bool = False, nof_repl_p_cand: int = 2
    ) -> List[str]:
        """
        Data augmentation with `ra`. It takes all the adjectives or nouns in a
        sentence `sent` (candidates) and replaces each of them `nof_repl_p_cand`
        times for a synonym. In particular, if an adjective/noun is selected, then
        the synonym/hyponym will be an adjective/noun.

        parameters:
            - sent (str) sentence to augment.
            - pos (str) POS to replace. Values: 'a' for adjectives, and 's' for
            nouns.
            - only_hyponyms (bool) only replace with hyponyms instead of synonyms.
            - nof_repl_p_cand (int) number of replacements per candidate.
        returns:
            - adv_examples (List[str]) list of augmented sentences.
        """
        adversarial_examples = []

        # Tokenize and pos tag sentence. TODO: the nltk tagger here is different to
        # the one used for other methods such as rsa ir aav (flair tagger) but keeps
        # the same notation. however wordnet uses a different pos notation. Since
        # synonyms and hyponyms are obtained with this last library following the
        # directions of https://hal.archives-ouvertes.fr/hal-02933266, this method
        # might be refactored to use only one POS notation.
        tokens = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(tokens)

        # 'a' and 's' in wordet is more or less equivalent to 'JJ' and 'NN' in
        # flair.
        if pos == "a" or pos == "s":
            TAG = "JJ"
        else:
            TAG = "NN"

        # Get all adjectives or adverbs and replace them by synonyms or hyponyms.
        words = [(idx, word) for idx, (word, tag) in enumerate(tags) if tag == TAG]
        for idx, word in words:
            candidates = get_synonyms(word, pos=pos, only_hyponyms=only_hyponyms)
            candidates = candidates[:nof_repl_p_cand]
            new_example = tokens.copy()

            for change in candidates:
                new_example[idx] = change
                adversarial_examples.append(" ".join(new_example))

        return adversarial_examples

    # TODO: cand_pos_name is not checked. So you can input a different POS than the
    # options we put, let's say NN and it will make undesirable behaviour!
    def add_adverbs_to_verbs(self, sent: str, cand_pos_name: str = "VB", nof_repl_p_cand: int = 2) -> List[str]:
        """
        Data augmentation with method `aav`. It selects all the words in `sent` that
        are POS labeled with `cand_pos_name`. We called these words as candidates.
        Each candidate is selected and new sentences are generated by adding an
        adverb before the candidate. In this case, speculative adverbs are used like
        certainly, likely, and clearly from a previously collected list.
        `cand_pos_name` can indicate verbs or adjectives. That is, the following
        values:

        - VB Verb, base form.
        - VBD Verb, past tense.
        - VBG Verb, gerund or present participle.
        - VBN Verb, past participle.
        - VBP Verb, non-3rd person singular present.
        - VBZ Verb, 3rd person singular present.
        - JJ, Adjective.

        parameters:
            - `sent` (str) sentence to augment.
            - `cand_pos_name` (str) adverb type. Values: 'VB', 'VBD', 'VBG', 'VBN',
            'VBP', 'VBZ', and 'JJ'.
            - `nof_repl_p_cand` (int) number of replacements per candidate. That is,
            per each candidate it creates `nof_repl_p_cand` sentences. For example,
            if `cand_pos_name = 'PER'`, `nof_repl_p_cand = 5` and `sent` has 2 named
            entities PER, `rne` generates 2 * 5 sentences.

        returns:
            - adv_examples (List[str]) list of augmented sentences.
        """
        # If adjectives are going to be modified, do not use `many` and `such`.
        BLACKLIST = ["many", "such"]

        # Create a flair instance Sentence.
        sent_ = Sentence(sent)

        # Tag sentence with POS.
        tagger_pos.predict(sent_)

        # Use speculative adverbs
        adv_list = speculative_list

        chosen_pos = []
        for label in sent_.get_labels("pos"):
            pos_type = label.value
            pos_text = label.data_point.text

            # Check the type of a tagged pos word and save them as candidates.
            if (pos_type == cand_pos_name) and (pos_text.lower() not in BLACKLIST):
                chosen_pos.append(label)

        # Randomly choose `nof_repl_p_cand` adverbs from `adv_list` and replace them
        # for each candidate.
        adv_examples = []
        for pos in chosen_pos:
            candidates = random.sample(adv_list, k=nof_repl_p_cand)

            examples = [
                add_exp_to_sent(  # noqa
                    sent=sent,
                    exp=change,
                    start_pos=pos.data_point.start_position,
                    end_pos=pos.data_point.end_position,
                    where="before",
                )
                for change in candidates
            ]
            adv_examples.extend(examples)

        return adv_examples

    def replace_in_domain_expressions(self, sent: str, nof_repl_p_cand: int = 2) -> List[str]:
        """
        Data augmentation with method `ri`. It replaces a list of manually-crafted
        expressions often used in HS messages (in-domain exp. not captured by `rne`)
        with other semantically similar expressions. That is, it checks all the
        ocurrences of in-domain expressions in `sent` (candidates), and generates
        `nof_repl_p_cand` sentences per each candidate by changing it for another
        manually-collected expression.

        parameters:
            - `sent` (str) sentence to augment.
            - `nof_repl_p_cand` (int) number of replacements per candidate.
        returns:
            adv_examples (List[str]) list of augmented examples.
        """
        adversarial_examples = []

        # Lower and tokenize sentence.
        sent = sent.lower()
        tokens = nltk.word_tokenize(sent)

        # Filter out ocurrences of in-domain expressions.
        words = [(idx, token) for idx, token in enumerate(tokens) if token in special_tokens.keys()]
        for idx, word in words:
            candidates = random.sample(special_tokens[word], k=min(len(special_tokens[word]), nof_repl_p_cand))
            for change in candidates:
                new_example = tokens.copy()
                new_example[idx] = change
                adversarial_examples.append(" ".join(new_example))

        return adversarial_examples

    def replace_scalar_adverbs(self, sent: str, nof_repl_p_cand: int = 2) -> List[str]:
        """
        Data augmentation with method `rsa`. It selects all the adverbs of pos type
        'RB' (candidates) in `sent`. For each candidate, it generates
        `nof_repl_p_cand` by replacing the candidate with an scalar adverb.

        parameters:
            - sent (str) sentence to augment.
            - nof_repl_p_cand (int) number of replacements per candidate.
        returns:
            - adv_examples (List[str]) list of augmented sentences.
        """
        # Create a flair instance Sentence.
        sent_ = Sentence(sent)

        # Tag sentence with POS.
        tagger_pos.predict(sent_)

        # Filter all adverbs in `sent`
        advs_in_sent = [label for label in sent_.get_labels("pos") if label.value == "RB"]

        # Randomly choose `nof_repl_p_cand` adverbs from `scalarity_list` and
        # replace them for each candidate.
        adv_examples = []
        for adv in advs_in_sent:
            # NOTE: scalarity_list contains adverbs that can go after or before an
            # adj/verb. In this case, we combine them in just one list.
            candidates = random.sample(scalarity_list["BEFORE"] + scalarity_list["AFTER"], k=nof_repl_p_cand)

            examples = [
                replace_exp_to_sent(
                    sent=sent, exp=change, start_pos=adv.data_point.start_position, end_pos=adv.data_point.end_position
                )
                for change in candidates
            ]
            adv_examples.extend(examples)
        return adv_examples

    def replace_named_entities(self, sent: str, cand_ner_name: str = "MISC", nof_repl_p_cand: int = 2) -> List[str]:
        """
        Data augmentation with method `rne`. It replaces a named entity (PER, LOC,
        ORG, and MISC) in the input sentence `sent`. A candidate NE in a sentence is
        replaced by another one according to a previously collected list of NEs.
        Then, the most similar NE is selected by using pre-trained FastText
        embeddings. Per each candidate it creates `nof_repl_p_cand` sentences. For
        example, if `ner_name = 'PER'`, `nof_repl_p_cand = 5` and `sent` has 2 named
        entities PER, `rne` generates 2 * 5 sentences.

        parameters:
            - `sent` (str) sentence to augment
            - `ner_name` (str) named entity. values `PER`, `LOC`, `ORG`, and `MISC`.
            - `nof_repl_p_cand` (int) number of replacements for named entity of
            `ner_name` type found.
        returns:
            - adv_examples (List[str]) list of augmented sentences.
        """
        # Create a flair instance Sentence.
        sent_ = Sentence(sent)

        # Tag sentence with NEs.
        tagger_ner.predict(sent_)

        adv_examples = []
        for label in sent_.get_labels("ner"):
            ner_type = label.value

            # Check if a tagged entity word is of type `ner_name`, that is, a
            # possible candidate.
            if ner_type == cand_ner_name:
                # Get the start and end positions of the entity.
                ner_text = label.data_point.text
                ner_start = label.data_point.start_position
                ner_end = label.data_point.end_position

                # Find FastText word embedding of the entity.
                w_vector = get_word_vector(ner_text, doc_emb=fasttext_doc_emb)

                # Get the most similar words to the entity in `vector_list`.
                candidates = find_closest_embeddings(vectors_list[ner_type], w_vector)  # noqa

                # Avoid replacement of the entity for itself.
                if ner_text in candidates:
                    candidates.remove(ner_text)

                # Make only `nof_repl_p_cand` replacements.
                candidates = candidates[:nof_repl_p_cand]

                # Create `nof_repl_p_cand` sentences by replacing the entity for the
                # most similar words found in the `vector_list`.
                examples = [
                    replace_exp_to_sent(sent=sent, exp=change, start_pos=ner_start, end_pos=ner_end)
                    for change in candidates
                ]

                # Collect all augmented sentences.
                adv_examples.extend(examples)
        return adv_examples
