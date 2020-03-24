from MeshTools.SplitManager import Splittables

n = 10

S = Splittables(n)
assert all(s.is_elementary for s in S)

S.split(1, (2, 3, 4))
S.split(3, (5, 6))
S.split(6, (8, 9))

for i, s in enumerate(S):
    if s.is_elementary:
        print(i, "*")
    else:
        print(i, ":", S.elementary_elements(i).array)
