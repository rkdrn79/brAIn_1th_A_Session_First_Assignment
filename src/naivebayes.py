import numpy as np

class NaiveBayesClassifier:
    def __init__(self, smoothing=1):

        """
        이 함수는 수정할 필요가 없습니다.
        """

        self.smoothing = smoothing
        self.epsilon = 1e-10  # 나눗셈 시 0으로 나누는 것을 방지하기 위한 작은 값


    def fit(self, x, y):

        """
        이 함수는 실제 학습(확률 계산)이 이루어지는 부분입니다.
        각 샘플의 특징 벡터가 self.data에 저장되고,
        각 샘플의 레이블이 self.labels에 저장됩니다.

        이후, 레이블 인덱스, 사전 확률, 가능도가 차례대로 계산됩니다.
        """

        self.data = x  # 입력된 데이터 저장 (샘플 x 특징 행렬)
        self.labels = y  # 입력된 레이블 저장 (샘플에 대한 정답 레이블)

        # 각 레이블에 속하는 샘플의 인덱스를 저장하는 딕셔너리 생성
        self.label_index = dict()
        self.label_name = set(self.labels)  # 레이블의 고유한 값들을 저장

        for lab in self.label_name:
            self.label_index[lab] = []

        for index, label in enumerate(self.labels):
            self.label_index[label].append(index)

        for lab in self.label_name:
            self.label_index[lab] = np.array(self.label_index[lab])

        # 사전 확률 계산
        self.get_prior()

        # 가능도 계산
        self.get_likelihood()

    def get_prior(self):

        """
        이 함수는 각 레이블의 사전 확률을 계산합니다.
        전체 샘플 중 해당 레이블의 샘플이 차지하는 비율로 계산됩니다.
        계산된 사전 확률은 self.prior에 저장됩니다.
        self.prior = {0: 0번 레이블에 대한 사전 확률, 1: 1번 레이블에 대한 사전 확률}
        """

        self.prior = dict()

        # 전체 샘플 수
        total_samples = len(self.labels)

        # TODO: 각 레이블에 대해 사전 확률을 계산
        # 예시: self.prior[label] = 해당 레이블에 속한 샘플 수 / 전체 샘플 수

        return self.prior

    def get_likelihood(self):

        """
        이 함수는 각 레이블에 대해 특징의 가능도를 계산합니다.
        특징 빈도수를 기반으로 라플라스 스무딩을 적용하여 가능도를 계산하고
        self.likelihood에 저장합니다.
        """

        self.likelihood = dict()

        # 총 특징의 수 (피처의 수)
        total_features = self.data.shape[1]

        # TODO: 각 레이블에 대해 계산
        # 1. 해당 레이블에 속하는 샘플들의 특징 값 합계 계산
        # 2. 라플라스 스무딩을 적용하여 특징의 가능도를 계산
        # 예시: smoothed_feature_count = feature_count + self.smoothing
        # 3. 계산된 가능도를 self.likelihood[label]에 저장

        return self.likelihood

    def get_posterior(self, x):

        """
        이 함수는 사전 확률(prior)과 가능도(likelihood)를 사용하여 사후 확률(posterior)을 계산합니다.
        로그를 사용해 계산하고, 오버플로우를 방지하기 위해 exp로 변환하여 확률을 구합니다.
        """

        self.posterior = []

        # TODO: 입력 데이터 x에 대해 각 샘플에 대해 사후 확률을 계산
        # 1. 각 레이블에 대해 log(사전 확률)와 log(가능도)를 계산
        # 2. 로그 확률을 더한 후 exp를 사용하여 확률로 변환
        # 3. 계산된 확률을 정규화하여 self.posterior에 저장

        return self.posterior

    def predict(self, x):

        """
        이 함수는 likelihood와 prior을 사용하여 실제 데이터를 예측합니다.
        사후 확률을 계산한 후, 가장 높은 확률의 클래스를 반환합니다.
        """

        posterior = self.get_posterior(x)
        return np.argmax(posterior, axis=1)