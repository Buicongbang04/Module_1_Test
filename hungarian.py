import random
from scipy.optimize import linprog
import numpy as np

def generate_random_integer_list(length, target_total):
    random_numbers = [random.random() for _ in range(length)]
    random_sum = sum(random_numbers)
    scaled_numbers = [int(num / random_sum * target_total) for num in random_numbers]
    
    # Điều chỉnh phần tử cuối để đảm bảo tổng đúng
    scaled_numbers[-1] += target_total - sum(scaled_numbers)
    return scaled_numbers

# Dữ liệu cung và cầu
supply = generate_random_integer_list(3, 1000)  # 3 nguồn cung với tổng là 25
demand = generate_random_integer_list(1000, 2731)  # 25 cửa hàng với tổng nhu cầu là 30

# Cân bằng cung và cầu
total_supply = sum(supply)
total_demand = sum(demand)

if total_supply < total_demand:
    # Thêm nguồn cung giả
    supply.append(total_demand - total_supply)
    dummy_supply = True
else:
    dummy_supply = False

if total_supply > total_demand:
    # Thêm điểm cầu giả
    demand.append(total_supply - total_demand)
    dummy_demand = True
else:
    dummy_demand = False

# Ma trận chi phí (bao gồm các dummy supply/demand nếu cần)
transpostCost = np.random.randint(100, 1000, size=(len(supply), len(demand)))
sellAbility = np.random.randint(10, 50, size=(len(demand)))
# Chuyển thành bài toán tối ưu hóa
num_supply = len(supply)
num_demand = len(demand)
cost_flat = ((transpostCost * 0.3)*(sellAbility * 0.7)).flatten()

# Ràng buộc tổng cung và cầu
A_eq = []
b_eq = []

# Ràng buộc nguồn cung
for i in range(num_supply):
    row = [1 if j // num_demand == i else 0 for j in range(num_supply * num_demand)]
    A_eq.append(row)
    b_eq.append(supply[i])

# Ràng buộc nhu cầu
for j in range(num_demand):
    row = [1 if j == k % num_demand else 0 for k in range(num_supply * num_demand)]
    A_eq.append(row)
    b_eq.append(demand[j])

# Lời giải
result = linprog(cost_flat, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")

# Kết quả
allocation = result.x.reshape((num_supply, num_demand))
allocation = np.clip(allocation, 0, None) 

if dummy_supply:
    allocation = allocation[:-1, :]
if dummy_demand:
    allocation = allocation[:, :-1]

print("Phân phối tối ưu:\n", allocation)
print("Tổng chi phí tối ưu:", result.fun)
