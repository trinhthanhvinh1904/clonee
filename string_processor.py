import Levenshtein

def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    lcs_length = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                lcs_length = max(lcs_length, dp[i][j])
            else:
                dp[i][j] = 0

    return lcs_length

def calculate_last_digit_score(ocr_result, student_id, similar_characters):
    score = 0
    for a, b in zip(ocr_result[-3:], student_id[-3:]):
        if a == b:
            score += 1
        elif a in similar_characters and b in similar_characters[a]:
            score += 0.5
    return score

def calculate_exact_match_score(ocr_result, student_id, similar_characters):
    temp_ocr = list(ocr_result)
    score = 0
    for char in student_id:
        if char in temp_ocr:
            score += 1
            temp_ocr.remove(char)
        elif char in similar_characters:
            for similar_char in similar_characters[char]:
                if similar_char in temp_ocr:
                    score += 0.5
                    temp_ocr.remove(similar_char)
                    break
    return score

#Import this function
def find_best_match(ocr_result, list_ids):
    similar_characters = {'&': '8', 'o': '0', 'O': '0', 'a': '0', 'S': '5', 'g': '9', 'b': '6'}

    best_match = None
    max_score = 0

    for id in list_ids:
        levenshtein_score = Levenshtein.ratio(ocr_result, id)
        last_digits_score = calculate_last_digit_score(ocr_result, id, similar_characters)
        exact_match_score = calculate_exact_match_score(ocr_result, id, similar_characters)
        lcs_length = longest_common_substring(ocr_result, id)

        combined_score = (
            levenshtein_score +
            0.1 * last_digits_score +
            0.1 * exact_match_score +
            0.1 * lcs_length
        )

        if combined_score > max_score:
            best_match = id
            max_score = combined_score

    return best_match, max_score

# student_ids = [
#     "22BI13001", "22BI13007", "22BI13008", "22BI13009", "22BI13012", "22BI13013", "22BI13015", 
#     "22BI13016", "22BI13018", "22BI13019", "22BI13021", "22BI13022", "22BI13023", "22BI13029", 
#     "22BI13032", "22BI13034", "22BI13037", "22BI13042", "22BI13043", "22BI13045", "22BI13047", 
#     "22BI13052", "22BI13055", "22BI13059", "22BI13068", "22BI13071", "22BI13077", "22BI13079", 
#     "22BI13081", "22BI13085", "22BI13088", "22BI13089", "22BI13092", "22BI13093", "22BI13094", 
#     "22BI13095", "22BI13096", "22BI13097", "22BI13098", "22BI13104", "22BI13106", "22BI13107", 
#     "22BI13115", "22BI13116", "22BI13117", "22BI13118", "22BI13122", "22BI13123", "22BI13126", 
#     "22BI13127", "22BI13128", "22BI13132", "22BI13136", "22BI13146", "22BI13147", "22BI13148", 
#     "22BI13149", "22BI13150", "22BI13154", "22BI13158", "22BI13161", "22BI13163", "22BI13170", 
#     "22BI13172", "22BI13181", "22BI13184", "22BI13186", "22BI13191", "22BI13201", "22BI13206", 
#     "22BI13211", "22BI13222", "22BI13240", "22BI13242", "22BI13244", "22BI13259", "22BI13261", 
#     "22BI13264", "22BI13274", "22BI13276", "22BI13278", "22BI13279", "22BI13282", "22BI13286", 
#     "22BI13288", "22BI13291", "22BI13301", "22BI13307", "22BI13308", "22BI13317", "22BI13318", 
#     "22BI13321", "22BI13323", "22BI13325", "22BI13328", "22BI13329", "22BI13331", "22BI13332", 
#     "22BI13338", "22BI13340", "22BI13342", "22BI13343", "22BI13344", "22BI13346", "22BI13356", 
#     "22BI13358", "22BI13367", "22BI13370", "22BI13371", "22BI13372", "22BI13374", "22BI13379", 
#     "22BI13380", "22BI13386", "22BI13390", "22BI13392", "22BI13393", "22BI13394", "22BI13400", 
#     "22BI13404", "22BI13406", "22BI13407", "22BI13410", "22BI13412", "22BI13419", "22BI13420", 
#     "22BI13426", "22BI13434", "22BI13435", "22BI13436", "22BI13438", "22BI13443", "22BI13446", 
#     "22BI13451", "22BI13453", "22BI13454", "22BI13463", "22BI13464", "22BI13474", "22BI13478", 
#     "22BI13479", "22BI13480", "22BI13482", "22BI13485", "BA12-003", "BA12-006", "BA12-007", 
#     "BA12-015", "BA12-016", "BA12-022", "BA12-034", "BA12-035", "BA12-045", "BA12-050", "BA12-057", 
#     "BA12-062", "BA12-068", "BA12-078", "BA12-082", "BA12-084", "BA12-088", "BA12-089", "BA12-090", 
#     "BA12-092", "BA12-093", "BA12-095", "BA12-102", "BA12-110", "BA12-126", "BA12-127", "BA12-128", 
#     "BA12-134", "BA12-138", "BA12-139", "BA12-150", "BA12-180", "BA12-189", "BA12-192", "BA12-193", 
#     "BI12-028", "BI12-079", "BI12-100", "BI12-123", "BI12-130", "BI12-144", "BI12-148", "BI12-183", 
#     "BI12-197", "BI12-210", "BI12-217", "BI12-233", "BI12-251", "BI12-252", "BI12-277", "BI12-300", 
#     "BI12-305", "BI12-336", "BI12-388", "BI11-183", "BI11-234"
# ]

# ocr_result = "B.A.12.13&"

# best_match, max_score = find_best_match(ocr_result, student_ids)

# print(f"OCR Output: {ocr_result}")
# print(f"Best match: {best_match}")
# print(f"Highest Combined Score: {max_score:.2f}")
