from pathlib import Path
import json


class HMM(object):
    '''隐式马尔科夫模型实现词性标注
    '''

    def __init__(self) -> None:
        self.__vocab = set()  # 词语集合
        self.__pos   = set()  # 词性集合

        # 概率矩阵
        self.__pi         = None
        self.__transition = None
        self.__emission   = None

        # 参数存放地址
        self.__params_dir = Path("./Params")
        self.__try_load()


    def __freq2prob(self, d):
        sum_freq = sum(d.values())
        return {p : freq / sum_freq for p, freq in d.items()}


    def __store(self):
        with (self.__params_dir / 'pi.json').open(mode='w') as f:
            json.dump(self.__pi, f, indent=2, ensure_ascii=False)

        with (self.__params_dir / 'transition.json').open(mode='w') as f:
            json.dump(self.__transition, f, indent=2, ensure_ascii=False)

        with (self.__params_dir / 'emission.json').open(mode='w') as f:
            json.dump(self.__emission, f, indent=2, ensure_ascii=False)


    def __try_load(self):
        pi_path = (self.__params_dir / 'pi.json')
        transition_path = (self.__params_dir / 'transition.json')
        emission_path = (self.__params_dir / 'emission.json')

        if pi_path.exists():
            with pi_path.open(mode="r") as f:
                self.__pi = json.load(f)
        if transition_path.exists():
            with transition_path.open(mode="r") as f:
                self.__transition = json.load(f)
        if emission_path.exists():
            with emission_path.open(mode="r") as f:
                self.__emission = json.load(f)


    def train(self, trainset_path):
        with Path(trainset_path).open(mode="r") as f:
            train_sentences = json.load(f)

        for sentences in train_sentences:
            for w, p in sentences:
                self.__vocab.add(w)
                self.__pos.add(p)

        # 频率
        pi_freq = {p : 0 for p in self.__pos}
        transition_freq = {p1 : {p2 : 0 for p2 in self.__pos} for p1 in self.__pos}
        emission_freq = {p : {w : 0 for w in self.__vocab} for p in self.__pos}

        # 开始训练
        for i, sentences in enumerate(train_sentences, start=1):
            print(f"training: {i}/{len(train_sentences)}...")
            # 学习初始概率，也就是句子第一个词性的概率
            pi_freq[sentences[0][1]] += 1

            # 状态对列表
            states_transition = [(pair1[1], pair2[1]) for pair1, pair2 in zip(sentences, sentences[1:])]

            # 学习转移概率
            for p1, p2 in states_transition:
                transition_freq[p1][p2] += 1

            # 学习发射概率
            for w, p in sentences:
                emission_freq[p][w] += 1

        # 计算出概率矩阵
        self.__pi = self.__freq2prob(pi_freq)
        self.__transition = {
            p : self.__freq2prob(freq_dis)
            for p, freq_dis in transition_freq.items()
        }
        self.__emission = {
            p : self.__freq2prob(freq_dis)
            for p, freq_dis in emission_freq.items()
        }
        self.__store()


    def __viterbi(self, sentence):
        N = len(self.__transition)         # 状态数
        T = len(sentence)                  # 观测序列长度
        states = self.__transition.keys()  # 词性列表

        viterbi_matrix = {                 # idx --> 词性 概率表
            i : {s : 0 for s in states}
            for i in range(0, T)
        }
        backpointers_matrix = {            # idx --> 词性 时, idx-1的词性
            i : {s : 0 for s in states}
            for i in range(0, T)
        }

        # 初始化边缘值
        for s in states:
            if sentence[0] in self.__emission[s]:
                viterbi_matrix[0][s] = self.__pi[s] * self.__emission[s][sentence[0]]
            backpointers_matrix[0][s] = 'start' # 句首记号

        def argmax(t, s):
            max_prob, argmax_pre_state = 0, None
            for i in states:
                p = viterbi_matrix[t-1][i] * self.__transition[i][s] * self.__emission[s][sentence[t]]
                if p > max_prob:
                    max_prob = p
                    argmax_pre_state = i
            return max_prob, argmax_pre_state

        # 递推
        for t in range(1, T):
            for s in states:
                viterbi_matrix[t][s], backpointers_matrix[t][s] = argmax(t,s)  # 第t个词, 选择词性s

        # 最后一个词的最大概率和相应状态
        max_prob, max_prob_final_state = 0, None
        for s in states:
            if viterbi_matrix[T-1][s] > max_prob:
                max_prob =  viterbi_matrix[T-1][s]
                max_prob_final_state = s

        # 回溯最佳路线
        best_path = [max_prob_final_state]
        for t in range(T-1, -1, -1):
            if best_path[-1] is None:  # 最后一个词max_prob=0的情形
                best_path.append(None)
            else:
                best_path.append(backpointers_matrix[t][best_path[-1]])
        best_path = list(reversed(best_path))

        return best_path


    def predict(self, sentence):
        path = self.__viterbi(sentence)
        s = f'<{path[0]}>'
        for w, p in zip(sentence, path[1:]):
            s += w + f'/<{p}> '
        return s


    def evaluate(self, testset_path):
        with Path(testset_path).open(mode="r") as f:
            test_data = json.load(f)
        total_tags, total_preds = list(), list()
        for i, t in enumerate(test_data, start=1):
            print(f"testing: {i}/{len(test_data)}...")
            sentence, tags = zip(*t)
            preds = self.__viterbi(sentence)
            total_tags.extend(tags)
            total_preds.extend(preds[1:])

        correct_num = sum([t == p for t, p in zip(total_tags,total_preds)])
        return correct_num / len(total_preds)


if __name__ == '__main__':
    MODE = 1
    hmm = HMM()

    if MODE == 0:
        print("开始训练...")
        hmm.train('./Dataset/pos_train.json')

    elif MODE == 1:
        print("开始预测...")
        sententence = [
            '上海', '浦东', '颁布', '了', '十', '件', '政策', '文件', '。'
        ]
        print(hmm.predict(sententence))

    elif MODE == 2:
        print("开始测试...")
        print(f"准确率: {hmm.evaluate('./Dataset/pos_test.json')}")
