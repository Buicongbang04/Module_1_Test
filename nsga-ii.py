# import numpy as np
# import pandas as pd
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.core.problem import Problem
# from pymoo.optimize import minimize
# from pymoo.termination import get_termination

# # Đọc dữ liệu từ file CSV (thay đường dẫn bằng file của bạn)
# df = pd.read_csv("Cua_hang_dien_thoai_1000.csv", encoding="utf-8-sig")

# # Lấy các cột dữ liệu cần thiết
# demand = df["Nhu cầu cửa hàng (số lượng điện thoại)"].values
# inventory = df["Số lượng hàng tồn kho"].values
# shipping_cost = df["Chi phí vận chuyển (VNĐ)"].values
# sales_avg = df["Số lượng bán ra trung bình (tháng)"].values
# vip_customers = df["Số lượng khách VIP"].values

# num_stores = len(demand)
# total_phones = 1000  # Tổng số lượng điện thoại cần phân phối
# brands = ["iPhone", "Samsung", "Xiaomi"]

# Định nghĩa bài toán tối ưu
# class PhoneDistributionProblem(Problem):
#     def __init__(self):
#         super().__init__(n_var=num_stores * len(brands),  # Biến quyết định: mỗi cửa hàng có 3 giá trị phân phối
#                          n_obj=3,  # 3 mục tiêu chính: chi phí vận chuyển, chênh lệch nhu cầu, tồn kho
#                          xl=0,     # Giới hạn dưới: không thể phân phối âm
#                          xu=total_phones // len(brands))  # Giới hạn trên cho mỗi loại điện thoại

#     def _evaluate(self, X, out, *args, **kwargs):
#         X = X.reshape(-1, num_stores, len(brands))  # Chuyển đổi thành mảng [số lượng giải pháp, cửa hàng, loại điện thoại]

#         # Tính tổng số lượng mỗi cửa hàng
#         total_distributed = np.sum(X, axis=2)

#         # Mục tiêu 1: Tối ưu chi phí vận chuyển
#         cost_obj = np.sum(X * shipping_cost[:, None], axis=(1, 2))

#         # Mục tiêu 2: Tối ưu chênh lệch nhu cầu
#         demand_obj = np.sum(np.abs(total_distributed - demand), axis=1)

#         # Mục tiêu 3: Giảm tồn kho
#         inventory_obj = np.sum(np.maximum(total_distributed - inventory, 0), axis=1)

#         out["F"] = np.column_stack([cost_obj, demand_obj, inventory_obj])

# # Cấu hình thuật toán NSGA-II
# problem = PhoneDistributionProblem()

# algorithm = NSGA2(pop_size=100)

# termination = get_termination("n_gen", 200)

# # Tối ưu hóa
# res = minimize(problem,
#                algorithm,
#                termination,
#                seed=1,
#                save_history=True,
#                verbose=True)

# # Lấy kết quả tối ưu nhất
# optimal_solutions = res.X
# optimal_objectives = res.F

# # Chuyển kết quả sang dạng dễ đọc
# best_solution = optimal_solutions[0].reshape(num_stores, len(brands))
# df["Phân phối iPhone"] = best_solution[:, 0].astype(int)
# df["Phân phối Samsung"] = best_solution[:, 1].astype(int)
# df["Phân phối Xiaomi"] = best_solution[:, 2].astype(int)
# Định nghĩa bài toán phân phối tổng số lượng điện thoại
# --------------------------------------------------------------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# from deap import base, creator, tools, algorithms

# # Đọc dữ liệu từ file CSV
# file_path = "Cua_hang_dien_thoai_1000.csv"  # Đường dẫn tới file CSV
# data = pd.read_csv(file_path)

# # Chuẩn bị dữ liệu
# num_stores = data.shape[0]
# demand = data["Nhu cầu cửa hàng (số lượng điện thoại)"].values
# inventory = data["Số lượng hàng tồn kho"].values
# transport_cost = data["Chi phí vận chuyển (VNĐ)"].values
# sales_avg = data["Số lượng bán ra trung bình (tháng)"].values
# vip_customers = data["Số lượng khách VIP"].values

# # Tổng số điện thoại cần phân phối
# total_phones = 1000

# # Hàm mục tiêu
# def evaluate(individual):
#     """
#     Hàm đánh giá một cá thể:
#     - f1: Chi phí vận chuyển (minimize).
#     - f2: Độ lệch giữa nhu cầu và phân phối (minimize).
#     - f3: Độ lệch tồn kho (minimize).
#     """
#     distribution = np.array(individual)
#     cost = np.sum(distribution * transport_cost)
#     demand_mismatch = np.sum(np.abs(distribution - demand))
#     inventory_mismatch = np.sum(np.abs(distribution - inventory))
#     return cost, demand_mismatch, inventory_mismatch

# # Cài đặt thuật toán di truyền với DEAP
# creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))  # Minimize tất cả
# creator.create("Individual", list, fitness=creator.FitnessMulti)

# toolbox = base.Toolbox()
# toolbox.register("attr_float", lambda: np.random.uniform(0, 1))  # Giá trị ngẫu nhiên ban đầu
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_stores)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# # Hàm tính toán fitness
# toolbox.register("evaluate", evaluate)

# # Hàm lai ghép, đột biến, chọn lọc
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
# toolbox.register("select", tools.selNSGA2)

# # Ràng buộc tổng số điện thoại phân phối
# def check_constraints(individual):
#     total = np.sum(individual)
#     if total != total_phones:  # Điều chỉnh để tổng = 1000
#         correction_factor = total_phones / total
#         individual = [max(0, min(demand[i], round(x * correction_factor))) for i, x in enumerate(individual)]
#     return individual

# toolbox.decorate("evaluate", tools.DeltaPenality(lambda ind: sum(ind) == total_phones, 1000))

# # Thực thi thuật toán
# population = toolbox.population(n=100)
# NGEN = 200
# CXPB, MUTPB = 0.7, 0.2

# for gen in range(NGEN):
#     offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
#     fits = map(toolbox.evaluate, offspring)
#     for fit, ind in zip(fits, offspring):
#         ind.fitness.values = fit
#     population = toolbox.select(offspring, len(population))

# # Lấy cá thể tốt nhất
# best_ind = tools.selBest(population, 1)[0]
# best_distribution = check_constraints(best_ind)

# # Lưu kết quả ra file
# output = pd.DataFrame({
#     "ID Cửa hàng": data["ID Cửa hàng"],
#     "Tên cửa hàng": data["Tên cửa hàng"],
#     "Phân phối tối ưu (số lượng điện thoại)": best_distribution
# })

# output.to_csv("Phan_phoi_toi_uu_DEAP.csv", index=False)
# print("Tổng là:", sum(best_distribution))
# print("Kết quả phân phối tối ưu đã được lưu vào file 'Phan_phoi_toi_uu_DEAP.csv'.")
# --------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random

random.seed(42)
np.random.seed(42)
# Đọc dữ liệu từ file CSV
file_path = "Cua_hang_dien_thoai_1000.csv"  # Đường dẫn tới file CSV
data = pd.read_csv(file_path)

# Chuẩn bị dữ liệu
num_stores = data.shape[0]
demand = data["Nhu cầu cửa hàng (số lượng điện thoại)"].values
inventory = data["Số lượng hàng tồn kho"].values
transport_cost = data["Chi phí vận chuyển (VNĐ)"].values
sales_avg = data["Số lượng bán ra trung bình (tháng)"].values
vip_customers = data["Số lượng khách VIP"].values

# Tổng số điện thoại cần phân phối
total_phones = 1000

# Hàm chuẩn hóa để tổng phân phối bằng 1000
def normalize_distribution(individual):
    """
    Hàm chuẩn hóa: đảm bảo tổng phân phối đúng bằng 1000 và không vượt nhu cầu.
    """
    total = sum(individual)
    if total == 0:
        return individual  # Nếu không có gì để phân phối, giữ nguyên
    factor = total_phones / total
    normalized = [max(0, min(demand[i], round(x * factor))) for i, x in enumerate(individual)]
    
    # Điều chỉnh lần cuối nếu vẫn chưa đúng tổng
    diff = total_phones - sum(normalized)
    if diff > 0:  # Nếu thiếu, phân phối thêm
        for i in np.argsort(demand - normalized):  # Ưu tiên cửa hàng thiếu nhiều nhất
            if diff == 0:
                break
            add = min(diff, demand[i] - normalized[i])
            normalized[i] += add
            diff -= add
    elif diff < 0:  # Nếu thừa, cắt bớt
        for i in np.argsort(normalized)[::-1]:  # Ưu tiên cửa hàng thừa nhiều nhất
            if diff == 0:
                break
            sub = min(-diff, normalized[i])
            normalized[i] -= sub
            diff += sub
    
    return normalized

# Hàm tính tối ưu mục tiêu
def evaluate(individual):
    """
    Hàm đánh giá một cá thể:
    - f1: Chi phí vận chuyển (minimize).
    - f2: Độ lệch giữa nhu cầu và phân phối (minimize).
    - f3: Độ lệch tồn kho (minimize).
    """
    distribution = np.array(normalize_distribution(individual))  # Chuẩn hóa trước
    cost = np.sum(distribution * transport_cost)
    demand_mismatch = np.sum(np.abs(distribution - demand))
    inventory_mismatch = np.sum(np.abs(distribution - inventory))
    return cost, demand_mismatch, inventory_mismatch
# Hàm tính mức độ ưu tiên
def preference_score(individual):
    """
    Hàm ưu tiên (chỉ tính khi tối ưu pha 2):
    - Ưu tiên cửa hàng bán nhiều (maximize).
    - Ưu tiên cửa hàng có nhiều khách VIP (maximize).
    """
    distribution = np.array(normalize_distribution(individual))
    sales_score = np.sum(distribution * sales_avg)
    vip_score = np.sum(distribution * vip_customers)
    return sales_score + vip_score

# Cài đặt DEAP
np.random.seed(42)  # Đặt seed cố định
random.seed(42)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))  # Minimize tất cả
creator.create("Individual", list, fitness=creator.FitnessMulti)
random.seed(42)
toolbox = base.Toolbox()
toolbox.register("attr_int", lambda: random.randint(0, 10))  # Giá trị ban đầu ngẫu nhiên
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_stores)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# Thực thi tối ưu
def optimize():
    random.seed(42)
    np.random.seed(42)
    population = toolbox.population(n=200)
    NGEN = 300
    CXPB, MUTPB = 0.7, 0.2

    for gen in range(NGEN):
        random.seed(42)
        np.random.seed(42)
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        for ind in offspring:
            ind[:] = normalize_distribution(ind)  # Chuẩn hóa mỗi lần đột biến hoặc lai ghép
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, len(population))

    # Lấy cá thể tốt nhất từ giai đoạn 1
    best_ind = tools.selBest(population, 1)[0]
    best_distribution = normalize_distribution(best_ind)

    # Giai đoạn 2: Tối ưu ưu tiên
    preference_scores = [preference_score(ind) for ind in population]
    best_pref_ind = population[np.argmax(preference_scores)]
    best_pref_distribution = normalize_distribution(best_pref_ind)

    return best_distribution, best_pref_distribution

# Thực thi
best_distribution, best_pref_distribution = optimize()

# Lưu kết quả ra file
output = pd.DataFrame({
    "ID Cửa hàng": data["ID Cửa hàng"],
    "Tên cửa hàng": data["Tên cửa hàng"],
    "Phân phối tối ưu (số lượng điện thoại)": best_pref_distribution
})

output.to_csv("Phan_phoi_toi_uu_DEAP.csv", index=False)
print("Kết quả phân phối tối ưu đã được lưu vào file 'Phan_phoi_toi_uu_DEAP.csv'.")
print("Tổng là:", sum(best_distribution))