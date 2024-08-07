# 나이 계산하는 함수
def user_age(year):
    return 2024-year

# 주민등록번호를 통해서 성별 파악하는 함수
def user_gender(min_data):
    if int(min_data[7]) % 2 == 0:
        return "여자"
    else:
        return "남자"