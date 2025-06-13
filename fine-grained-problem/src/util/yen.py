import copy

def Dijkstra(network,s,d):#Dijkstra calculate shortest path of s-d and return the path and cost
    #print("Start Dijstra Path……")
    path=[]# shortest path of s-d
    n=len(network)# adjacency matrix dimension, that is, the number of nodes
    fmax=9999999
    w=[[0 for i in range(n)]for j in range(n)]# convert adjacency matrix to dimension matrix, that is, 0→max
    book=[0 for i in range(n)]# whether it is the smallest mark list
    dis=[fmax for i in range(n)]# the minimum distance from s to other nodes
    book[s-1]=1# node index starts from 1, list index starts from 0
    midpath=[-1 for i in range(n)]# last hop list
    u=s-1
    for i in range(n):
        for j in range(n):
            if network[i][j]!=0:
                w[i][j]=network[i][j]#0→max
            else:
                w[i][j]=fmax
            if i==s-1 and network[i][j]!=0:# the minimum distance to directly connected nodes is network[i][j]
                dis[j]=network[i][j]
    for i in range(n-1):# n-1 times traversal, except for s node
        min=fmax
        for j in range(n):
            if book[j]==0 and dis[j]<min:# if not traversed and distance is minimum
                min=dis[j]
                u=j
        book[u]=1
        for v in range(n):# u's directly connected nodes traversal
            if dis[v]>dis[u]+w[u][v]:
                dis[v]=dis[u]+w[u][v]
                midpath[v]=u+1# last hop update
    j=d-1# j is the index
    path.append(d)# because the last hop is stored, the destination node d is added first, and then reversed
    while(midpath[j]!=-1):
        path.append(midpath[j])
        j=midpath[j]-1
    path.append(s)
    path.reverse()# reverse the list
    #print(path)
    #print(midpath)
    #print(dis)
    return path

def return_path_sum(network,path):
    result=0
    for i in range(len(path)-1):
        result+=network[path[i]-1][path[i+1]-1]
    return result

def add_limit(path,s):#path=[[[1,3,4,6],5],[[1,3,5,6],7],[[1,2,4,6],8]
    result=[]
    for item in path:
        if s in item[0]:
            result.append([s,item[0][item[0].index(s)+1]])
    result=[list(r) for r in list(set([tuple(t) for t in result]))]#remove duplicates
    return result

def return_shortest_path_with_limit(network,s,d,limit_segment,choice):#limit_segment=[[3,5],[3,4]]
    mid_net=copy.deepcopy(network)
    for item in limit_segment:
        mid_net[item[0]-1][item[1]-1]=mid_net[item[1]-1][item[0]-1]=0
    s_index=choice.index(s)
    for point in choice[:s_index]:# the points before s are disabled
        for i in range(len(mid_net)):
            mid_net[point-1][i]=mid_net[i][point-1]=0
    mid_path=Dijkstra(mid_net,s,d)
    return mid_path

def judge_path_legal(network,path):
    for i in range(len(path)-1):
        if network[path[i]-1][path[i+1]-1]==0:
            return False
    return True

def k_shortest_path(network,s,d,k):
    k_path=[]# result list
    alter_path=[]# alternative list
    kk=Dijkstra(network,s,d)
    k_path.append([kk,return_path_sum(network,kk)])
    while(True):
        if len(k_path)==k:break
        choice=k_path[-1][0]
        for i in range(len(choice)-1):
            limit_path=[[choice[i],choice[i+1]]]#limit selected path
            if len(k_path)!=1:
                limit_path.extend(add_limit(k_path[:-1],choice[i]))
            mid_path=choice[:i]
            mid_res=return_shortest_path_with_limit(network,choice[i],d,limit_path,choice)
            if judge_path_legal(network,mid_res):
                mid_path.extend(mid_res)
            else:
                continue
            mid_item=[mid_path,return_path_sum(network,mid_path)]
            if mid_item not in k_path and mid_item not in alter_path:
                alter_path.append(mid_item)
        if len(alter_path)==0:
            print("total only {} paths".format(len(k_path)))
            return k_path
        alter_path.sort(key=lambda x:x[-1])
        x=alter_path[0][-1]
        y=len(alter_path[0][0])
        u=0
        for i in range(len(alter_path)):
            if alter_path[i][-1]!=x:
                break
            if len(alter_path[i][0])<y:
                y=len(alter_path[i][0])
                u=i
        k_path.append(alter_path[u])
        alter_path.pop(u)
    # for item in k_path:
    #     print(item)
    return k_path
if __name__=='__main__':
    network=[[0,3,2,0,0,0],
            [3,0,1,4,0,0],
            [2,1,0,2,3,0],
            [0,4,2,0,2,1],
            [0,0,3,2,0,2],
            [0,0,0,1,2,0]]
    network = [[1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],
               [1,2,3,4,5,6,7,8,9,10],]
    k_shortest_path(network,1,6,10)