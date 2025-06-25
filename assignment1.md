# Assignment #1: 虚拟机，Shell & 大语言模型

Updated 2309 GMT+8 Feb 20, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



**作业的各项评分细则及对应的得分**

| 标准                                 | 等级                                                         | 得分 |
| ------------------------------------ | ------------------------------------------------------------ | ---- |
| 按时提交                             | 完全按时提交：1分<br/>提交有请假说明：0.5分<br/>未提交：0分  | 1 分 |
| 源码、耗时（可选）、解题思路（可选） | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于2个：0分 | 1 分 |
| AC代码截图                           | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于：0分 | 1 分 |
| 清晰头像、PDF文件、MD/DOC附件        | 包含清晰的Canvas头像、PDF文件以及MD或DOC格式的附件：1分<br/>缺少上述三项中的任意一项：0.5分<br/>缺失两项或以上：0分 | 1 分 |
| 学习总结和个人收获                   | 提交了学习总结和个人收获：1分<br/>未提交学习总结或内容不详：0分 | 1 分 |
| 总得分： 5                           | 总分满分：5分                                                |      |
>
> 
>
> **说明：**
>
> 1. **解题与记录：**
>       - 对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>    
>2. **课程平台与提交安排：**
> 
>   - 我们的课程网站位于Canvas平台（https://pku.instructure.com ）。该平台将在第2周选课结束后正式启用。在平台启用前，请先完成作业并将作业妥善保存。待Canvas平台激活后，再上传你的作业。
> 
>       - 提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>3. **延迟提交：**
> 
>   - 如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
> 
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：

加完了约分

代码：

```python
a,b,c,d=map(int,input().split())
e=a*d+b*c
f=b*d
def gcd(x,y):
    while True:
        if x == 0 or y == 0:
            return x + y
        else:
            if x<y:
                y=y%x
            else:
                x=x%y
print(f'{e//gcd(e,f)}/{f//gcd(e,f)}')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\garys\Pictures\Screenshots\屏幕截图 2025-03-02 194650.png)



### 1760.袋子里最少数目的球

 https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/




思路：

简单的二分查找

代码：

```python
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        left = 1
        right = max(nums)
        while left < right:
            mid = (left + right) // 2
            total_ops = 0
            for x in nums:
                total_ops += (x - 1) // mid
                if total_ops > maxOperations:
                    break
            if total_ops <= maxOperations:
                right = mid
            else:
                left = mid + 1
        return left
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250311004652382](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311004652382.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135



思路：

先设定查找范围，然后利用贪心策略判断在给定的最大开销下是否能合理划分月份

代码：

```python
def isPossible(days, M, x):
    count = 1
    current_sum = 0
    for day in days:
        current_sum += day
        if current_sum > x:
            count += 1
            current_sum = day
    return count <= M

def findMinMaxExpense(days, M):
    left = max(days)
    right = sum(days)
    while left < right:
        mid = (left + right) // 2
        if isPossible(days, M, mid):
            right = mid
        else:
            left = mid + 1
    return left

N, M = map(int, input().split())
days = [int(input()) for _ in range(N)]
print(findMinMaxExpense(days, M))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250311013042404](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311013042404.png)



### 27300: 模型整理

http://cs101.openjudge.cn/practice/27300/



思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### Q5. 大语言模型（LLM）部署与测试

本任务旨在本地环境或通过云虚拟机（如 https://clab.pku.edu.cn/ 提供的资源）部署大语言模型（LLM）并进行测试。用户界面方面，可以选择使用图形界面工具如 https://lmstudio.ai 或命令行界面如 https://www.ollama.com 来完成部署工作。

测试内容包括选择若干编程题目，确保这些题目能够在所部署的LLM上得到正确解答，并通过所有相关的测试用例（即状态为Accepted）。选题应来源于在线判题平台，例如 OpenJudge、Codeforces、LeetCode 或洛谷等，同时需注意避免与已找到的AI接受题目重复。已有的AI接受题目列表可参考以下链接：
https://github.com/GMyhf/2025spring-cs201/blob/main/AI_accepted_locally.md

请提供你的最新进展情况，包括任何关键步骤的截图以及遇到的问题和解决方案。这将有助于全面了解项目的推进状态，并为进一步的工作提供参考。

本人的电脑选购考量见总结，电脑配置为英特尔最新一代Ultra 5 225H，核显是Arc-130T，居然可以运行14B和32B，但由于14B较慢且有一些傻，我又配了一个Deepseek Coder v2 Lite Instruct，16B的且运行速度很快，下面包含对这两个大模型的测试。

![image-20250311011856885](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311011856885.png)

我先测试了下作业题，发现14B的解题速度很慢，但是是对的，只不过用时远多于正常水平。

![image-20250311011416902](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311011416902.png)

You

给你一个整数数组 nums ，其中 nums[i] 表示第 i 个袋子里球的数目。同时给你一个整数 maxOperations 。

你可以进行如下操作至多 maxOperations 次：

选择任意一个袋子，并将袋子里的球分到 2 个新的袋子中，每个袋子里都有 正整数 个球。 比方说，一个袋子里有 5 个球，你可以把它们分到两个新袋子里，分别有 1 个和 4 个球，或者分别有 2 个和 3 个球。 你的开销是单个袋子里球数目的 最大值 ，你想要 最小化 开销。

请你返回进行上述操作后的最小开销。

class Solution: def minimumSize(self, nums: List[int], maxOperations: int) -> int:

示例 1：

输入：nums = [9], maxOperations = 2 输出：3 解释：

- 将装有 9 个球的袋子分成装有 6 个和 3 个球的袋子。[9] -> [6,3] 。
- 将装有 6 个球的袋子分成装有 3 个和 3 个球的袋子。[6,3] -> [3,3,3] 。 装有最多球的袋子里装有 3 个球，所以开销为 3 并返回 3 。 示例 2：

输入：nums = [2,4,8,2], maxOperations = 4 输出：2 解释：

- 将装有 8 个球的袋子分成装有 4 个和 4 个球的袋子。[2,4,8,2] -> [2,4,4,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,4,4,4,2] -> [2,2,2,4,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,2,2,4,4,2] -> [2,2,2,2,2,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,2,2,2,2,4,2] -> [2,2,2,2,2,2,2,2] 。 装有最多球的袋子里装有 2 个球，所以开销为 2 并返回 2 。 示例 3：

输入：nums = [7,17], maxOperations = 2 输出：7

提示：

1 <= nums.length <= 105 1 <= maxOperations, nums[i] <= 109



Assistant

deepseek-r1-distill-qwen-14b

![img](file:///C:/Users/garys/AppData/Local/Programs/LM%20Studio/resources/app/.webpack/renderer/cba4276f67b6138b115854c21d97b7cc.png)

Thoughts



Thought for 20 minutes 4 seconds

哦，这个问题看起来有点难度。我得仔细读题目，然后想办法找出解决办法。

那么问题大意是说，我们有一个整数数组nums，每个元素代表一个袋子里的球的数量。我们可以进行最多maxOperations次操作，每次操作选一个袋子，把它分成两个新袋子，每个都有至少一个球。我们的目标是在所有可能的操作之后，让最大值尽可能小。然后要返回这个最小的最大值。

举个例子，比如示例1中的情况：nums是[9]，maxOperations是2。第一次操作把9分成6和3，然后把6分成3和3，这样最终得到的袋子都是3或更少，所以开销是3。

那怎么才能找到这个最小的开销呢？我觉得这可能与二分查找有关，因为我们需要找一个临界值，并且判断在给定操作次数下是否可以达到这个值。

首先，我应该考虑用什么方法来确定最小的最大值。二分法是一个不错的选择。我们可以对可能的最大值进行二分，然后检查是否在这个假设的最大值下，使用最多maxOperations次操作后是可以实现的。

那么具体的步骤可能是这样的：

1. 找出所有袋子的最小可能最大值的范围。比如左边界可以是1（因为至少每个袋子有一个球），右边界可以是数组中最大的数。或者更准确地说，初始的上界应该是数组中的最大元素，这样我们可以在二分的时候缩小范围。
2. 对于每一个中间值mid，在这个假设下，计算需要多少次操作才能将所有大于等于mid的袋子分割到不超过mid。如果总的操作次数小于等于maxOperations，则可能可以找到更小的mid；反之则需要增大mid。

但是这里有一个关键点：如何计算每个数至少需要多少次操作才能分解到都不超过mid？

比如，对于一个数x来说，在每次操作中，我们可以把它分成两个部分，同时每一步的最大值不能超过mid。那其实，我们需要将这个x分解成尽可能多的较小的数。但是要满足最终所有袋子里的数量都不超过mid。

举个例子，假设mid是3，而有一个袋子有9球。那么第一次分，比如可以分成6和3；然后对6再分，分成3和3；总共两次操作，得到三个3，这样总的最大值变成了3。这正是示例1的情况。

那具体来说，如何计算每个数需要多少次操作？假设当前mid是目标最大值，那么对于数组中的每一个元素num：

我们需要将num分解成若干个<=mid的球堆。每次操作只能拆分成两部分，并且每一步都必须满足至少分成两个正整数部分。这实际上是一个递归的过程吗？

或者可以这样想：每个num要分解为k个<=mid的部分，那么最少需要多少次操作？例如，如果num <= mid，那么不需要分解；否则的话，我们需要拆分它。

比如，假设我们要将一个num拆分成m个部分，那么每次拆分都会增加一个新的袋子。这其实类似于二叉树的结构，每一步都将一个节点分成两个子节点。所以，总的拆分次数等于总生成的新袋子数减一。或者更准确地说，初始时有一个袋子，经过k次操作后会得到k+1个袋子。

那对于一个num来说，当num <= mid的时候，不需要拆分；否则，我们需要计算这个num能被拆分成多少个<=mid的部分，这样就可以算出需要的最少的操作次数。

那么，具体怎么计算拆分次数呢？比如，我们想让最终所有袋子都不超过mid。那么每个数必须分解到尽可能多的不超过mid的堆里，同时每次只能拆分成两个部分。

这个过程其实类似于把num分成尽可能小的堆，并且每一步只能分割一次。但是这可能有点复杂。

或者换个思路：对于一个给定的mid，我们能计算出将所有大于mid的数分解到只含有<=mid的数所需的最少操作次数总和是否小于等于maxOperations。如果是的话，说明这个mid是可行的，我们可以尝试寻找更小的mid；否则，我们需要增大mid。

那如何计算每个num需要多少次拆分呢？

假设我们有一个函数f(num, mid)返回将num分解到所有袋子都<=mid所需要的最小操作次数。那么总的操作次数就是数组中每个数的f值之和。如果这个总和<=maxOperations，则说明mid是可以接受的。

那问题转化为，如何计算f(num, mid)？

对于num来说：

- 如果num <= mid：不需要拆分，返回0。
- 否则：我们需要将num分成两个部分，且每部分都尽可能小。但实际上，每次操作只能生成一个额外的袋子。那么，为了得到最多的袋子（从而减少最大值），我们应该尽量多次拆分较大的数。

或者更直接地说，f(num, mid)等于分解次数，而这个次数可以通过不断地除以2来计算吗？比如，num需要被分割成多少次才能都<=mid？

例如，假设mid=3：

对于num=9，第一次拆分成6和3：操作一次。但此时的6还需要再拆分，所以总的操作数是两次。

或者，另一个例子，num=7，mid=5：

- 7>5 → 拆分成3和4 → 操作次数+1。
- 现在有两个堆：3（<=5，不需要处理）和4（<=5，不需要处理）。所以总操作数是1次。

那对于更复杂的num来说，比如num=8，mid=3：

需要拆分几次？

第一次拆分成3和5 → 5>3。然后对5进行拆分：拆分成2和3 → 操作次数+1。此时总操作次数为两次？或者这可能不是最优的。

哦，或者更高效的拆分方式是，尽量将num分解成尽可能多的小堆，这样可以减少总的拆分次数吗？

或者，对于f(num, mid)，计算所需拆分次数的方法应该是：每次尽可能多地分割出mid个球。比如，假设mid=3：

num = 9 → 可以直接拆分成3和6？不，这可能不是最优的，因为6还是大于3。那是不是需要多次拆分？

或者另一个思路：f(num, mid)的最小操作次数等于将num分解为k个部分，每个部分<=mid的情况下，最少需要多少次分割。而每次分割只能将一个堆分成两个，所以总分割次数等于初始1个减去最终k个？或者说，分割的总次数是 (k-1)，其中k是分解后的袋子数？

这可能是一个关键点。

比如：

对于num=9，mid=3：需要拆分成三个3。那么每次只能拆分一个袋子。第一次把9拆分成6和3 → 袋子数目增加到2。然后对6进行拆分：拆分为3和3 → 袋子数目变为3。总操作次数是2次。

所以，这个过程的拆分次数等于分解后的袋子数减去1。例如，分解为k个袋子需要k-1次分割吗？

如果是这样的话，那么对于num来说，分解后最少能有多少个袋子？这取决于mid的值。比如，最大的k是当每个堆尽可能小的时候得到的，也就是当mid是最大的情况下。

或者更准确地说：我们希望将num分解成尽可能多的部分，每部分最多mid个球。这样，所需的最小拆分次数就是分解后的堆数减去1。

那如何计算分解后最少能有多少个袋子？这可以通过不断地除法来得到。

比如：

对于num和mid来说，最大的k是 ceil(num / mid)。但是，当ceil的结果大于等于2时，我们需要进行一次分割，并且每次分割都会将一个堆拆分成两个较小的堆。

或者，更直接地说：分解后的最小堆数是 ceil(num / mid)。而这个过程需要多少次操作呢？

因为每次操作只能将一堆分成两堆，所以总的操作次数等于 (ceil(num/mid) - 1) 次。例如：

- num=9, mid=3 → ceil(9/3)=3堆，操作次数为2次。
- num=6, mid=3 → ceil(6/3)=2，操作次数是1次。
- num=5, mid=3 → ceil(5/3)=2，操作次数是1次。

这似乎是对的。所以，对于num来说：

如果 num <= mid → 0 操作 否则： k = (num + mid -1) // mid  # 这样计算ceil(num/mid) operations += (k-1)

这样就得到了每个数需要的操作次数。然后，将所有数的这个值加起来，看看是否<= maxOperations。

那我们的目标是找到最小的mid，使得sum(f(num, mid)) <= maxOperations.

那么现在，问题转化为如何用二分查找法来寻找这样的最小mid。

接下来，确定二分的范围：

- 左边界left=1（因为至少每个袋子有一个球）
- 右边界right=max(nums)（如果不需要拆分的话）

但是，当max(nums)被拆分多次后，可能会变得更小。例如，在示例2中，初始数组是[2,4,8,2], maxOperations=4：

最大值是8，但通过4次操作将其分解到每个袋子最多2个球。

所以，我们的right应该设为max(num)，这样在二分的时候能够覆盖可能的范围。因为如果mid低于当前最小的可能值，那么sum(f)会变得太大，从而无法满足条件。

现在，具体步骤如下：

1. 计算数组中的最大数作为初始右边界。
2. 使用二分查找法，在left和right之间寻找最小的mid。
3. 对于每一个mid，计算所有num需要的操作次数总和。如果sum <= maxOperations，则可能可以尝试更小的mid；否则，需要增大mid。

那让我们来看示例3的情况：

输入：nums = [7, 17], maxOperations=2

那么，假设我们想找到mid=7是否可行？

对于7来说：

- 第一个数是7，不需要拆分。
- 第二个数是17。ceil(17/7)=3 → 需要操作次数是3-1=2次。

总操作次数为0+2=2 <= maxOperations（2），所以mid=7可行吗？那是否可以得到更小的mid？

比如，尝试mid=6：

计算每个数的操作次数： num=7：ceil(7/6)=2 → 操作次数是1。 num=17: ceil(17/6)=3→ 17/(6) = 2.833,所以向上取整为3？不，等一下，17除以6等于2余5，所以ceil(17/6)=3。操作次数是2次。

总操作次数是1+2=3 > maxOperations（2）。那mid=6不可行。因此，在这种情况下，mid必须至少是7才能满足条件。那么结果就是7，如示例3所示。

这说明我们的思路是对的：当sum(f(num, mid)) <= maxOperations时，我们可以尝试更小的mid；否则，增大mid。

那么现在的问题是如何实现这个过程。

接下来，我们需要编写一个函数，计算给定mid的情况下，总需要多少操作次数。然后在二分过程中调用这个函数。

那么，现在具体怎么写代码呢？

首先，初始化left=1，right=max(nums).

然后循环进行二分：

while left < right:

```sql
mid = (left + right) // 2

计算sum_ops：所有num>mid的情况下的(k-1)之和，其中k是ceil(num/mid)

if sum_ops <= maxOperations:

    right = mid  # 因为要找最小的mid，所以尝试更小的值

else:

    left = mid + 1
```



最后返回left或者right。

那这个方法是否正确？让我们再看几个例子：

示例2的情况：nums=[2,4,8,2], maxOperations=4.

初始max是8. 我们寻找mid，使得sum_ops<=4.

比如，假设mid=2:

计算每个num的操作次数：

对于2:不需要拆分→0操作。

4: ceil(4/2)=2 → 操作1次。总操作+1

8:ceil(8/2)=4 → 操作3次。总操作+3

另一个2：0.

sum_ops=1+3=4，刚好等于maxOperations.所以mid=2是可行的。

那是否可以更小？比如mid=1？

不，因为每个数拆分成至少两个堆，需要更多的操作次数。

所以最终结果返回mid=2。

这与示例2的结果一致。

再看示例3的情况：

nums=[7,17], maxOperations=2.

当mid=7时，sum_ops是0 (7不需要拆分) + 2次（17需要拆分成3个堆），总和是2<=maxOperations。所以我们可以尝试找更小的mid？比如6。

当mid=6时：

计算每个数的操作次数：

7: ceil(7/6)=2 →操作1次.

17:ceil(17/6)=3→ 操作两次.

sum_ops=1+2=3>2.所以不能取mid=6。因此，只能取7或更大的值。

这样，最终结果就是7。

综上所述，我们的思路是正确的。

那么，在编写代码时，需要注意：

1. 如何计算ceil(num/mid)？可以用 (num + mid -1) // mid.

比如，7和mid=6：(7+5)//6 = 12//6=2 → ceil(7/6)=2.

而8和mid=3：(8+2)/3=10/3=3 → ceil(8/3)=3。

这很正确。

所以，在代码中，遍历数组中的每一个num，计算k=(num + mid -1)//mid。然后将k-1加到sum_ops中。

那么，现在编写这个函数：

在Python中，可以写一个辅助函数：

def calculate_operations(nums, mid): total_ops = 0 for num in nums: if num > mid: k = (num + mid -1) // mid total_ops += (k-1) # 超过maxOperations的情况要提前剪枝吗？ return total_ops

那么，在二分查找的过程中，调用这个函数来计算总的操作次数。

接下来，处理边界条件。比如当mid为0的时候？这可能不会发生，因为left初始是1。

另外，考虑到nums的长度可能很大（比如1e5），所以要确保代码的效率。每个二分步骤的时间复杂度是O(n)，而n=1e5，最多log(max_num)次二分。假设max_num是1e9，那么log次数大约30次左右。总时间复杂度是30*1e5=3e6，这应该没问题。

那现在，编写代码：

初始化left = 1，right = max(nums).

然后循环直到left < right:

```sql
mid = (left + right) //2

ops = calculate_operations(nums, mid)

if ops <= maxOperations:

    right = mid

else:

    left = mid +1
```



返回left.

这应该可以解决问题。

那么测试一下示例情况：

比如，示例3的情况：

nums=[7,17],maxOperations=2.

计算mid=(left+right)//2的初始值：left=1,right=17 → mid=9.

sum_ops = (7>9？否→0) + (17>9？是→ceil(17/9)=2 → 2-1=1. sum_ops=1 <=2 → 所以right=mid=9。

接下来，left=1, right=9.

循环继续：

mid=(1+9)//2=5.

计算sum_ops:

num=7>5：是→ceil(7/5)=2 → 操作1次.

num=17>5: yes. ceil(17/5)=4→操作3次.

总和为4 > maxOperations 2 → 所以left=mid+1=6.

现在，left=6, right=9.

mid=(6+9)//2 =7.

计算sum_ops:

num=7>7？否。0.

num=17>7: yes. ceil(17/7)=3 →操作2次. sum=2<=2→right=7.

现在，left=6, right=7.

mid=(6+7)//2=6.

计算sum_ops:

num=7>6→ yes. ceil(7/6)=2 →操作1.

num=17:ceil(17/6)=3 →操作2次.

总和是3>2→left=6+1=7.

现在，left=7, right=7→循环结束，返回7。正确。

综上所述，算法是正确的。

那么，编写Python代码：

读取nums数组，maxOperations。初始化left和right，然后二分查找，并计算sum_ops.

现在，将这个思路写成一个函数：

class Solution: def minimumSize(self, nums: List[int], maxOperations: int) -> int: left = 1 right = max(nums)

```sql
    while left < right:
        mid = (left + right) // 2
        total_ops = 0
        
        for num in nums:
            if num > mid:
                k = (num + mid -1) // mid
                total_ops += (k-1)
                
                # 如果当前total_ops已经超过了maxOperations，可以提前终止循环
                if total_ops > maxOperations:
                    break
        
        if total_ops <= maxOperations:
            right = mid
        else:
            left = mid + 1
    
    return left
```



这应该可以解决问题。

另外，注意到在for循环中，如果total_ops已经超过了maxOperations，可以提前break，以减少计算时间。这对n=1e5时可能会有帮助，但每次都要计算完整个数组，所以可能不会省太多时间。

综上所述，以上代码是正确的。

为了解决这个问题，我们需要找到一种方法来最小化在最多 `maxOperations` 次操作后袋子中球的最大数量。通过将每个袋子的球分成两个新的袋子，我们可以逐步减少最大值。

### 方法思路

1. **问题分析**:
   - 我们需要进行最多 `maxOperations` 次操作。
   - 每次操作可以选择一个袋子，并将其分成两个正整数的新袋子。
   - 目标是最小化最终的最大球数。
2. **直觉和见解**:
   - 使用二分查找来确定最小的最大值。我们可以通过调整中间值 `mid` 来找到满足条件的最小最大值。
   - 对于每个可能的最大值 `mid`，计算所需的总操作次数。如果操作次数不超过 `maxOperations`，则可以尝试更小的 `mid`。
3. **算法选择和优化**:
   - 使用二分查找来高效地缩小范围，检查每个中间值 `mid` 是否满足条件。
   - 对于每个 `mid`，计算每个袋子需要拆分成多少次操作，并累加总操作次数。如果总操作次数在允许范围内，则继续尝试更小的值。

### 解决代码

```python
from typing import List

class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        left = 1
        right = max(nums)
        
        while left < right:
            mid = (left + right) // 2
            total_ops = 0
            
            for num in nums:
                if num > mid:
                    k = (num + mid - 1) // mid
                    total_ops += (k - 1)
                    
                    if total_ops > maxOperations:
                        break
            
            if total_ops <= maxOperations:
                right = mid
            else:
                left = mid + 1
        
        return left
```



### 代码解释

- **初始化边界**: `left` 设为 1，`right` 设为数组中的最大值。
- **二分查找**: 在每次循环中计算中间值 `mid`，并检查每个袋子在拆分到 `mid` 次数时的操作次数。
- **操作次数计算**: 对于每个大于 `mid` 的袋子，计算需要拆分成多少次，并累加到总操作次数中。如果总操作次数超过允许范围，提前终止循环。
- **调整边界**: 根据总操作次数是否在允许范围内调整二分查找的边界，直到找到最小的最大值。

这种方法确保了我们能够在对数时间内找到最优解，时间复杂度为 O(n log(max_value))，其中 n 是数组长度，max_value 是数组中的最大值。

3.20 tok/sec

•

4457 tokens

•

29.03s to first token

•

Stop reason: EOS Token Found

同样的题目我喂给coder秒出答案。这让我好奇是不是这些题已经被训练过了？

![image-20250311011756163](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311011756163.png)

You

给你一个整数数组 nums ，其中 nums[i] 表示第 i 个袋子里球的数目。同时给你一个整数 maxOperations 。

你可以进行如下操作至多 maxOperations 次：

选择任意一个袋子，并将袋子里的球分到 2 个新的袋子中，每个袋子里都有 正整数 个球。 比方说，一个袋子里有 5 个球，你可以把它们分到两个新袋子里，分别有 1 个和 4 个球，或者分别有 2 个和 3 个球。 你的开销是单个袋子里球数目的 最大值 ，你想要 最小化 开销。

请你返回进行上述操作后的最小开销。

class Solution: def minimumSize(self, nums: List[int], maxOperations: int) -> int:

示例 1：

输入：nums = [9], maxOperations = 2 输出：3 解释：

- 将装有 9 个球的袋子分成装有 6 个和 3 个球的袋子。[9] -> [6,3] 。
- 将装有 6 个球的袋子分成装有 3 个和 3 个球的袋子。[6,3] -> [3,3,3] 。 装有最多球的袋子里装有 3 个球，所以开销为 3 并返回 3 。 示例 2：

输入：nums = [2,4,8,2], maxOperations = 4 输出：2 解释：

- 将装有 8 个球的袋子分成装有 4 个和 4 个球的袋子。[2,4,8,2] -> [2,4,4,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,4,4,4,2] -> [2,2,2,4,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,2,2,4,4,2] -> [2,2,2,2,2,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,2,2,2,2,4,2] -> [2,2,2,2,2,2,2,2] 。 装有最多球的袋子里装有 2 个球，所以开销为 2 并返回 2 。 示例 3：

输入：nums = [7,17], maxOperations = 2 输出：7

提示：

1 <= nums.length <= 105 1 <= maxOperations, nums[i] <= 109



Assistant

deepseek-coder-v2-lite-instruct

![img](file:///C:/Users/garys/AppData/Local/Programs/LM%20Studio/resources/app/.webpack/renderer/cba4276f67b6138b115854c21d97b7cc.png)

为了找到进行上述操作后的最小开销，我们可以使用二分查找的方法。具体来说，我们需要找到一个合适的阈值（即每个袋子最终可以拥有的最大球数），然后通过计算需要进行的操作次数来验证这个阈值是否可行。

以下是详细的步骤：

1. **定义目标**：我们的目标是找到一个阈值 `threshold`，使得在进行了最多 `maxOperations` 次操作后，所有袋子中的最大球数不超过这个阈值。
2. **二分查找**：我们使用二分查找来确定这个阈值。初始的左边界是1，右边界是数组中最大的元素（因为无论如何分割，单个袋子的最大球数不会超过数组中的最大元素）。
3. **验证阈值**：对于每个候选的阈值，我们计算需要进行的操作次数。如果操作次数不超过 `maxOperations`，则这个阈值是可行的。
4. **计算操作次数**：对于每个候选的阈值，我们遍历数组中的所有元素，计算将这些元素分割到满足阈值需要的操作次数。

下面是实现上述逻辑的代码：

```python
from typing import List

class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        left, right = 1, max(nums)
        
        while left < right:
            mid = (left + right) // 2
            operations = sum((num - 1) // mid for num in nums)
            
            if operations > maxOperations:
                left = mid + 1
            else:
                right = mid
        
        return left
```



### 解释代码：

- **二分查找**：我们使用 `left` 和 `right` 来表示当前的搜索范围。初始时，`left` 是1，`right` 是数组中的最大值。
- **中间值计算**：每次迭代中，我们计算 `mid` 作为当前的候选阈值。
- **操作次数计算**：对于每个候选阈值 `mid`，我们计算需要进行的操作次数 `operations`，即遍历所有袋子，计算将每个袋子中的球数分割到满足 `mid` 需要的操作次数之和。
- **调整搜索范围**：如果当前的 `operations` 超过了 `maxOperations`，说明阈值太小了，我们需要增大左边界 `left`；否则，我们可以尝试减小右边界 `right` 以找到更小的可行阈值。

通过这种方式，我们可以在较短的时间内找到最小的开销。

21.26 tok/sec

•

585 tokens

•

31.53s to first token

•

Stop reason: EOS Token Found



**由于14B太慢了而且有过半个小时做不出来一道题的情况所以剩下的题目我都用coder跑**

[1745. 分割回文串 IV](https://leetcode.cn/problems/palindrome-partitioning-iv/)

![image-20250311012506275](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311012506275.png)

You

困难 相关标签 相关企业 提示 给你一个字符串 s ，如果可以将它分割成三个 非空 回文子字符串，那么返回 true ，否则返回 false 。

当一个字符串正着读和反着读是一模一样的，就称其为 回文字符串 。

示例 1：

输入：s = "abcbdd" 输出：true 解释："abcbdd" = "a" + "bcb" + "dd"，三个子字符串都是回文的。 示例 2：

输入：s = "bcbddxy" 输出：false 解释：s 没办法被分割成 3 个回文子字符串。

提示：

3 <= s.length <= 2000 s 只包含小写英文字母。 class Solution: def checkPartitioning(self, s: str) -> bool:



Assistant

deepseek-coder-v2-lite-instruct

![img](file:///C:/Users/garys/AppData/Local/Programs/LM%20Studio/resources/app/.webpack/renderer/cba4276f67b6138b115854c21d97b7cc.png)

要解决这个问题，我们需要检查字符串 `s` 是否可以被分割成三个非空的回文子字符串。我们可以使用动态规划来识别所有的回文子字符串，然后检查是否有满足条件的分割方法。

以下是详细的步骤和代码实现：

1. **预处理**：首先，我们需要一个函数来检测一个子字符串是否是回文。我们可以使用动态规划来构建这个表。
2. **分割检查**：然后，我们遍历所有可能的分割点，检查是否可以将字符串分成三个回文子字符串。

以下是具体的代码实现：

```python
class Solution:
    def checkPartitioning(self, s: str) -> bool:
        n = len(s)
        
        # 预处理，构建一个表来存储子字符串是否为回文
        dp = [[False] * n for _ in range(n)]
        
        # 所有长度为1的子字符串都是回文
        for i in range(n):
            dp[i][i] = True
        
        # 检查长度为2到n的所有子字符串
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    if length == 2:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
        
        # 检查是否可以分割成三个回文子字符串
        for i in range(1, n - 1):
            for j in range(i, n - 1):
                if dp[0][i - 1] and dp[i][j] and dp[j + 1][n - 1]:
                    return True
        
        return False
```



### 解释：

1. **预处理**：
   - `dp` 是一个二维列表，`dp[i][j]` 表示子字符串 `s[i:j+1]` 是否是回文。
   - 初始化所有长度为1的子字符串（即对角线）为 `True`，因为单个字符总是回文。
   - 然后检查所有长度为2到n的子字符串，更新 `dp` 表来标记回文。
2. **分割检查**：
   - 遍历所有可能的分割点 `i` 和 `j`，确保 `i` 在第一个分割点和第二个分割点之间，`j` 在第二个分割点和最后一个字符之间。
   - 如果 `dp[0][i-1]`, `dp[i][j]`, 和 `dp[j+1][n-1]` 都是回文，则返回 `True`。
   - 如果没有找到这样的分割点，返回 `False`。

这个方法通过动态规划有效地识别了所有可能的回文子字符串，并检查了是否可以将其分成三个非空的回文子字符串。

23.34 tok/sec

•

722 tokens

•

8.80s to first token

•

Stop reason: EOS Token Found

[41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)

![image-20250311012757959](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311012757959.png)

You

困难 相关标签 相关企业 提示 给你一个字符串 s ，如果可以将它分割成三个 非空 回文子字符串，那么返回 true ，否则返回 false 。

当一个字符串正着读和反着读是一模一样的，就称其为 回文字符串 。

示例 1：

输入：s = "abcbdd" 输出：true 解释："abcbdd" = "a" + "bcb" + "dd"，三个子字符串都是回文的。 示例 2：

输入：s = "bcbddxy" 输出：false 解释：s 没办法被分割成 3 个回文子字符串。

提示：

3 <= s.length <= 2000 s 只包含小写英文字母。 class Solution: def checkPartitioning(self, s: str) -> bool:



Assistant

deepseek-coder-v2-lite-instruct

![img](file:///C:/Users/garys/AppData/Local/Programs/LM%20Studio/resources/app/.webpack/renderer/cba4276f67b6138b115854c21d97b7cc.png)

要解决这个问题，我们需要检查字符串 `s` 是否可以被分割成三个非空的回文子字符串。我们可以使用动态规划来识别所有的回文子字符串，然后检查是否有满足条件的分割方法。

以下是详细的步骤和代码实现：

1. **预处理**：首先，我们需要一个函数来检测一个子字符串是否是回文。我们可以使用动态规划来构建这个表。
2. **分割检查**：然后，我们遍历所有可能的分割点，检查是否可以将字符串分成三个回文子字符串。

以下是具体的代码实现：

```python
class Solution:
    def checkPartitioning(self, s: str) -> bool:
        n = len(s)
        
        # 预处理，构建一个表来存储子字符串是否为回文
        dp = [[False] * n for _ in range(n)]
        
        # 所有长度为1的子字符串都是回文
        for i in range(n):
            dp[i][i] = True
        
        # 检查长度为2到n的所有子字符串
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    if length == 2:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
        
        # 检查是否可以分割成三个回文子字符串
        for i in range(1, n - 1):
            for j in range(i, n - 1):
                if dp[0][i - 1] and dp[i][j] and dp[j + 1][n - 1]:
                    return True
        
        return False
```



### 解释：

1. **预处理**：
   - `dp` 是一个二维列表，`dp[i][j]` 表示子字符串 `s[i:j+1]` 是否是回文。
   - 初始化所有长度为1的子字符串（即对角线）为 `True`，因为单个字符总是回文。
   - 然后检查所有长度为2到n的子字符串，更新 `dp` 表来标记回文。
2. **分割检查**：
   - 遍历所有可能的分割点 `i` 和 `j`，确保 `i` 在第一个分割点和第二个分割点之间，`j` 在第二个分割点和最后一个字符之间。
   - 如果 `dp[0][i-1]`, `dp[i][j]`, 和 `dp[j+1][n-1]` 都是回文，则返回 `True`。
   - 如果没有找到这样的分割点，返回 `False`。

这个方法通过动态规划有效地识别了所有可能的回文子字符串，并检查了是否可以将其分成三个非空的回文子字符串。

23.34 tok/sec

•

722 tokens

•

8.80s to first token

•

Stop reason: EOS Token Found



You

困难  相关标签 相关企业  提示   给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。 请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。  示例 1： 输入：nums = [1,2,0] 输出：3 解释：范围 [1,2] 中的数字都在数组中。 示例 2： 输入：nums = [3,4,-1,1] 输出：2 解释：1 在数组中，但 2 没有。 示例 3： 输入：nums = [7,8,9,11,12] 输出：1 解释：最小的正数 1 没有出现。  提示： • 1 <= nums.length <= 105 • -231 <= nums[i] <= 231 - 1 class Solution: def firstMissingPositive(self, nums: List[int]) -> int:



Assistant

deepseek-coder-v2-lite-instruct

![img](file:///C:/Users/garys/AppData/Local/Programs/LM%20Studio/resources/app/.webpack/renderer/cba4276f67b6138b115854c21d97b7cc.png)

要解决这个问题，我们需要找到数组中未出现的最小正整数。由于时间复杂度要求为 O(n)*O*(*n*) 且只使用常数级别额外空间，我们可以利用数组的索引位置来标记数字的存在与否。具体步骤如下：

1. **将所有负数和大于数组长度的数变为一个特殊值**：我们选择用数组中的最大值加一来表示这个特殊值（例如，数组长度为 n*n*，则用 n+1*n*+1 表示）。这样可以确保这些数字不会影响我们的结果。
2. **将每个数字放在正确的位置上**：遍历数组，如果当前数字在 [1,n][1,*n*] 范围内，则将其放到索引为该数字减一的位置上（例如，数字 33 放在索引 22 处）。
3. **寻找未出现的最小正整数**：再次遍历数组，找到第一个位置不正确的数字，即为未出现的最小正整数。如果所有位置都正确，则返回 n+1*n*+1。

以下是实现代码：

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Step 1: Place each number in its correct position (if possible)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        
        # Step 2: Find the first missing positive integer
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        # If all positions are filled correctly, return n+1
        return n + 1
```



### 解释：

- **第一步**：我们遍历数组，对于每个数字 nums[i]*n**u**m**s*[*i*]，如果它在 [1,n][1,*n*] 范围内且不在正确的位置上（即 `nums[nums[i] - 1] != nums[i]`），则将其交换到正确的位置。这样可以确保每个数字都在其应该在的位置上。
- **第二步**：再次遍历数组，找到第一个不满足 `nums[i] == i + 1` 的位置，返回该位置加一即为未出现的最小正整数。如果所有位置都正确，则返回 n+1*n*+1。

这种方法的时间复杂度为 O(n)*O*(*n*)，因为我们最多遍历两次数组，且每次交换操作都是常数时间复杂度的。空间复杂度为 O(1)*O*(1)，因为我们没有使用额外的空间来存储数据结构。

19.81 tok/sec

•

646 tokens

•

39.45s to first token

•

Stop reason: EOS Token Found

**总结：**coder真的好强，基本上1秒生成20个token，这还是核显的情况下，它将对我练习编程有很大帮助！

### Q6. 阅读《Build a Large Language Model (From Scratch)》第一章

作者：Sebastian Raschka

请整理你的学习笔记。这应该包括但不限于对第一章核心概念的理解、重要术语的解释、你认为特别有趣或具有挑战性的内容，以及任何你可能有的疑问或反思。通过这种方式，不仅能巩固你自己的学习成果，也能帮助他人更好地理解这一部分内容。





## 2. 学习总结和个人收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

老师好！作为一名数学学院的学生，我曾深入钻研数学竞赛，对逻辑与解题有着独特的热爱，未来也立志投身量化领域。遗憾的是，上学期由于数院的军理课与课程时间冲突，我错过了选修您的课程的机会（或许这也是许多数院同学的无奈）。不过，这学期我决心全力以赴，追赶进度。 假期期间，我的电脑突发故障，恰逢国补政策的契机，我决定直接更换新设备。由于家乡的国补政策不如北京优惠，且新一代英特尔处理器（Arrow Lake）即将发布，我经过权衡，最终在上一周拿到了心仪的电脑。目前，50系显卡尚未问世，而40系显卡又稍显过时，因此我选择了轻薄本作为过渡设备。 如今，我已经迅速适应了新电脑，并投入到紧张的学习中。目前我正在挑选计算机概论的题目练习。得益于数学竞赛的背景，我对算法的理解速度很快，但代码实现时常常会犯一些小错误。不过，在大模型的辅助下，我的进步速度明显加快，我也对未来的学习充满信心。
