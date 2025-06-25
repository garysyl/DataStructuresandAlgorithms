# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Complied by 尚昀霖，数学科学学院



> **说明：**
>
> 1. **⽉考**：AC6 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：



代码：

```python
n,k=map(int,input().split())
a=[]
b=k-1
c=[]
for i in range(1,n+1):
    a.append(i)
for _ in range(n-1):
    c.append(a[b])
    a.remove(a[b])
    b=(b+(k-1))%len(a)
print(*c, sep=' ')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408162334465](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250408162334465.png)



### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：



代码：

```python
N,K=map(int,input().split())
a=[]
b=[]
for _ in range(N):
    a.append(int(input()))
right=max(a)
left=0
while True:
    mid=(right+left+1)//2
    if right==mid:
        break
    sum = 0
    for i in a:
        sum+=i//mid
    if sum<K:
        right=mid
    else:
        left=mid
print(left)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408163756313](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250408163756313.png)



### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：



代码：

```python
import sys
from collections import deque
n = int(sys.stdin.readline())
result = []
for _ in range(n):
    line = sys.stdin.readline().strip()
    if not line:
        continue
    tokens = line.split()
    nodes = []
    for i in range(0, len(tokens), 2):
        char = tokens[i]
        degree = int(tokens[i + 1])
        nodes.append((char, degree))
    if not nodes:
        continue
    children = [[] for _ in range(len(nodes))]
    pointer = 0
    q = deque()
    q.append(0)
    while q:
        p = q.popleft()
        d = nodes[p][1]
        if d == 0:
            continue
        start = pointer + 1
        end = start + d - 1
        if end >= len(nodes):
            break
        for i in range(start, end + 1):
            children[p].append(i)
        q.extend(range(start, end + 1))
        pointer = end
    def dfs(p):
        for child in children[p]:
            dfs(child)
        result.append(nodes[p][0])
    dfs(0)
print(' '.join(result))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408165341946](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250408165341946.png)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：



代码：

```python
T=int(input())
a=list(map(int,input().split()))
a.sort()
left=0
right=len(a)-1
t=a[0]+a[len(a)-1]
while left<right:
    if abs(a[left] + a[right] - T) < abs(T - t) or abs(T - a[left] - a[right]) == t - T:
        t = a[left] + a[right]
    if a[left]+a[right]<=T:
        left+=1
    elif a[left]+a[right]>T:
        right-=1
print(t)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408214948007](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250408214948007.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：



代码：

```python
T=int(input())
a=[]
for _ in range(T):
    a.append(int(input()))
b=max(a)
c=(b-2)//10
d=[]
e=int(b**(1/2))
for i in range(1,c+1):
    d.append(i*10+1)
for j in range(2,e):
    for k in d:
        if k%j==0 and k!=j:
            d.remove(k)
for _ in range(T):
    g=[]
    print(f'Case{_+1}:')
    f=a[_]
    if f<=11:
        print('NULL')
    else:
        for i in d:
            if i>=f:
                break
            else:
                g.append(i)
        print(*g, sep=' ')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408165741735](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250408165741735.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：



代码：

```python
m = int(input())
submissions = []
for _ in range(m):
    submission = input().split(',')
    submissions.append((submission[0], submission[1], submission[2]))

team_data = {}
for team, problem, result in submissions:
    if team not in team_data:
        team_data[team] = {'correct': set(), 'submissions': 0}
    team_data[team]['submissions'] += 1
    if result == 'yes' and problem not in team_data[team]['correct']:
        team_data[team]['correct'].add(problem)

teams = []
for team, data in team_data.items():
    teams.append((team, len(data['correct']), data['submissions']))

teams.sort(key=lambda x: (-x[1], x[2], x[0]))

ranked_teams = teams[:12]
for i, (team, correct, submissions) in enumerate(ranked_teams):
    print(f"{i + 1} {team} {correct} {submissions}")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250408170004663](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250408170004663.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

月考艰难ak，这次题水了，但还是有小半的题目是靠数学功力硬撑的，数算还是不熟。忙完期中之后接着做题。









