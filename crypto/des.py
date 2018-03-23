import json

import flask
import numpy as np


class Des:
    # Длина ключа и сообщения.
    LENGTH = 8

    # Начальная перестановка.
    INITIAL_PERMUTATION = [
        58, 50, 42, 34, 26, 18, 10, 2,
        60, 52, 44, 36, 28, 20, 12, 4,
        62, 54, 46, 38, 30, 22, 14, 6,
        64, 56, 48, 40, 32, 24, 16, 8,
        57, 49, 41, 33, 25, 17, 9, 1,
        59, 51, 43, 35, 27, 19, 11, 3,
        61, 53, 45, 37, 29, 21, 13, 5,
        63, 55, 47, 39, 31, 23, 15, 7
    ]

    # Перестановка с расширением в ячейке Фейстеля.
    EXPAND_PERMUTATION = [
        32, 1, 2, 3, 4, 5,
        4, 5, 6, 7, 8, 9,
        8, 9, 10, 11, 12, 13,
        12, 13, 14, 15, 16, 17,
        16, 17, 18, 19, 20, 21,
        20, 21, 22, 23, 24, 25,
        24, 25, 26, 27, 28, 29,
        28, 29, 30, 31, 32, 1
    ]

    # Р-перестановка (перемешивающая) в ячейке Фейстеля.
    P_PERMUTATION = [
        16, 7, 20, 21,
        29, 12, 28, 17,
        1, 15, 23, 26,
        5, 18, 31, 10,
        2, 8, 24, 14,
        32, 27, 3, 9,
        19, 13, 30, 6,
        22, 11, 4, 25
    ]

    # Завершающая перестановка.
    TERMINATE_PERMUTATION = [
        40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25
    ]

    # Нелинейный компонент - S-блоки (все в одной структуре данных).
    S_REPLACEMENTS = [
        [
            [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
            [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
            [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
            [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
        ],
        [
            [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
            [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
        ],
        [
            [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
            [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
            [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
        ],
        [
            [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
            [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
            [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
        ],
        [
            [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
            [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
            [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
        ],
        [
            [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
            [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
        ],
        [
            [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
            [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
        ],
        [
            [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
        ]
    ]

    # Номера раундов (начиная с 1), в которых циклический сдвиг при генерации ключа должен осуществляться на 1 бит.
    SH1_ROUNDS = [1, 2, 3, 9, 16]

    # Матрица перестановки раундового ключа РС1.
    PC1 = [
        57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4
    ]

    # Матрица перестановки раундового ключа РС1.
    PC2 = [
        14, 17, 11, 24, 1, 5,
        3, 28, 15, 6, 21, 10,
        23, 19, 12, 4, 26, 8,
        16, 7, 27, 20, 13, 2,
        41, 52, 31, 37, 47, 55,
        30, 40, 51, 45, 33, 48,
        44, 49, 39, 56, 34, 53,
        46, 42, 50, 36, 29, 32
    ]

    @staticmethod
    def split_to_pair(x: np.array) -> tuple:
        """
        Разделяет массив бит на 2 половины.
        :param x:
        :return:
        """

        return np.array(x[:len(x) // 2]), np.array(x[len(x) // 2:])

    @staticmethod
    def permutation(x: np.array, rule: list) -> np.array:
        """
        Осуществляет перестановку бит массива в соответствии с переданным правилом.
        :param x:    Массив для перестановки бит.
        :param rule: Правило перестановки (массив с новым порядком бит, заданным их номерами начиная с 1).
        :return:
        """

        return np.array([x[i - 1] for i in rule])

    @staticmethod
    def s_change(x: np.array, s_rule: list) -> int:
        """
        Осуществляет замену в соответстии с таблицей замены S-блока.
        :param x:      Массив для замены.
        :param s_rule: Правило замены - двумерный массив с правилом работы S-блока.
        :return:
        """

        '''
        Вычисление индекса строки и столбца правила замены в S-блоке (с 0):
        1) для получения из заданного шестибитного слова индекса строки, необходимо выбрать нулевой и последний биты,
        объединить их в двухбитовую строку (нулевой бит старше) и
        интерпретировать это как двухбитовое число;
        2) для получения индекса столбца необходимо выбрать биты с 1 по предпоследний включительно и интерпретировать их
        как число от 0 до 15.
        '''
        line, col = (x[0] << 1) | x[-1], int(''.join([str(x) for x in x[1:-1]]), 2)
        return s_rule[line][col]

    @staticmethod
    def feistel(r: np.array, round_key: np.array) -> np.array:
        """
        Повторяет работу функции Фейстеля.
        :param r:
        :param round_key:
        :return:
        """

        # XOR раундового ключа и результата расширенной перестановки над правой половиной блока.
        r_48 = np.bitwise_xor(Des.permutation(r, Des.EXPAND_PERMUTATION), round_key)

        # Результат воздействия S-блоков.
        r_s = np.array([Des.s_change(r, s_rule) for r, s_rule in zip(r_48.reshape(8, -1), Des.S_REPLACEMENTS)])

        # P-перестановки.
        return Des.permutation(Des.to_bit_array(r_s, 4), Des.P_PERMUTATION)

    @staticmethod
    def get_round_key(key: np.array, round_number: int) -> np.array:
        """
        Формирует рауновый ключ.
        :param key:
        :param round_number:
        :return:
        """

        if round_number not in range(1, 17):
            return 0

        # Перестановка РС1 и разделение на левую и правую части.
        l, r = Des.split_to_pair(Des.permutation(key, Des.PC1))

        # Сдвиг частей ключа.
        for round_count in range(1, round_number + 1):
            sh = [-2, -1][round_count in Des.SH1_ROUNDS]
            l, r = np.roll(l, sh), np.roll(r, sh)

        # Перестановка РС2 над целым ключом.
        return Des.permutation(np.concatenate((l, r)), Des.PC2)

    @staticmethod
    def str_to_bit_array(s: str) -> np.array:
        """
        Преобразует строку в массив бит.
        :param s:
        :return:
        """

        return Des.to_bit_array([ord(char) for char in s])

    @staticmethod
    def to_bit_array(a, length=LENGTH) -> np.array:
        """
        Преобразует массив чисел в массив бит.
        :param a:
        :param length:
        :return:
        """

        bin_s = ['{0:b}'.format(a) for a in a]
        return np.array([[int(x) for x in '0' * (length - len(bin_s)) + bin_s] for bin_s in bin_s]).flatten()

    @staticmethod
    def bit_array_to_str(x: np.array) -> str:
        """
        Преобразует массив бит в строку.
        :param x:
        :return:
        """

        return ''.join([chr(symbol_code) for symbol_code in [int(''.join([str(bit) for bit in byte]), 2)
                                                             for byte in x.reshape(8, -1)]])

    @staticmethod
    def des_encrypt(m: np.array, k: np.array, n_rounds: int) -> np.array:
        """
        Осуществляет шифрование в соответствии с классическим алгоритмом DES.
        :param m:        Открытый текст.
        :param k:        Секретный ключ.
        :param n_rounds: Количество раундов шифрования.
        :return:
        """

        bin_key = {
            'bin_key': m,
        }

        print('Зашифрование текста')
        print('Открытый текст {} ({})'.format(Des.bit_array_to_str(m), m))
        print('Ключ {} ({})'.format(Des.bit_array_to_str(k), k))

        print('\nНачало процедуры зашифрования')

        # Начальная перестановка и разделение на две части.
        l, r = Des.split_to_pair(Des.permutation(m, Des.INITIAL_PERMUTATION))

        print('Начальная перестановка {}'.format(np.concatenate((l, r))))

        start_swap = '{}'.format(np.concatenate((l, r)))

        print('Сеть Фейстеля на {} раундов'.format(n_rounds))

        list_rounds = []

        # Сеть Фейстеля на заданное число раундов.
        for round_count in range(1, n_rounds + 1):
            print('     Раунд {}'.format(round_count))
            l, r = r, np.bitwise_xor(l, Des.feistel(r, Des.get_round_key(k, round_count)))
            print('     Результат раунда {}'.format(np.concatenate((l, r))))
            round_data = {
                'num_round': round_count,
                'res': '{}'.format(np.concatenate((l, r)))
            }
            list_rounds.append(round_data)

        print('Состояние перед финальной перестановкой {}'.format(np.concatenate((l, r))))

        state_before_result_swap = '{}'.format(np.concatenate((l, r)))

        # Финальная перестановка и разделение на две части для изменения порядка их следования.
        l, r = Des.split_to_pair(Des.permutation(np.concatenate((l, r)), Des.TERMINATE_PERMUTATION))

        print('Финальная перестановка {}'.format(np.concatenate((l, r))))

        final_swap = '{}'.format(np.concatenate((l, r)))

        print('Изменение порядка следования левой и правой частей {}'.format(np.concatenate((r, l))))

        change_order_left_right = '{}'.format(np.concatenate((r, l)))

        print('Результат {}'.format(Des.bit_array_to_str(np.concatenate((r, l)))))

        # Объединения правой и левой частей.
        res = {
            'start_swap': start_swap,
            'rounds': list_rounds,
            'result': '{}'.format(Des.bit_array_to_str(np.concatenate((r, l)))),
            'state_before_result_swap': state_before_result_swap,
            'final_swap': final_swap,
            'change_order_left_right': change_order_left_right
        }
        return flask.jsonify(**res)

    @staticmethod
    def des_decrypt(m: np.array, k: np.array, n_rounds: int) -> np.array:
        """
        Осуществляет расшифрование в соответствии с классическим алгоритмом DES.
        :param m:        Открытый текст.
        :param k:        Секретный ключ.
        :param n_rounds: Количество раундов шифрования.
        :return:
        """

        l, r = Des.split_to_pair(m)

        # Нейтрализация перестановки левой и правой частей, осуществленной при шифровании,
        # начальная перестановка и разделение на две части.
        l, r = Des.split_to_pair(Des.permutation(np.concatenate((r, l)), Des.INITIAL_PERMUTATION))

        # Обратная сеть Фейстеля на заданное число раундов.
        for round_count in range(n_rounds, 0, -1):
            l, r = np.bitwise_xor(r, Des.feistel(l, Des.get_round_key(k, round_count))), l

        # Объединния частей и финальная перестановка.
        return Des.permutation(np.concatenate((l, r)), Des.TERMINATE_PERMUTATION)
