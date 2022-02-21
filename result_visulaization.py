#----------------------------
# Model Result Visulizer
# By: ziang chen
#----------------------------


#### Create Series Pics for Predicts Results


from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm import tqdm
import pickle
from config import conf

conz=conf()

avm = ArgoverseMap()
data_file=open(conz.model+ 'model_output.pll','rb')
result=pickle.load(data_file)

class id_points:
    def __init__(self):
        self.current_xy=dict()
        self.previous_xy=dict()

    def update(self,id,xy):
        if id in self.current_xy.keys():
            self.previous_xy[id]=self.current_xy[id]
        self.current_xy[id]=xy

    def draw(self):
        for id in self.current_xy.keys():
            try:
                plt.scatter(self.current_xy[id][0][0],self.current_xy[id][0][1],c='blue' )
            except:
                pass

def city_name_from_data(data):
    return data['CITY_NAME'].values[0]

def draw_map(data,N):
    map_fig= plt.figure(0,figsize=(10,10))
    num=0
    cityname=city_name_from_data(data)
    for id_ in data['TRACK_ID']:
        xy= data[data['TRACK_ID']  ==  id_ ][['X','Y']].values
        lane_ids=avm.get_lane_ids_in_xy_bbox(xy[0,0],xy[0,1],city_name=cityname,query_search_range_manhattan=200)
        for id in lane_ids:
        # Draw from lane id
            avm.draw_lane(id,cityname)
        num+=1
        if num>=N:
            break

def draw_candidates(data):
    avid=data[data['OBJECT_TYPE']=='AGENT']['TRACK_ID'].values[0]
    cityname=city_name_from_data(data)
    try:
        lane_ids=avm.get_candidate_centerlines_for_traj(data[data['TRACK_ID']  ==  avid ][['X','Y']].values,city_name=cityname)
        for lane in lane_ids:
            plt.plot(lane[:,0],lane[:,1],'k--')
        plt.xlim(  min(min(lane[:,0]),min(data['X'])),
                   max(max(lane[:,0]) ,  max(data['X'])  ) )
        plt.ylim(  min(min(lane[:,1]),min(data['Y'])),
                   max(max(lane[:,1]) ,  max(data['Y'])  ) )
    except:
        pass

def draw_timed_agents(data,t):
    """
    idxy= data[data['TIMESTAMP']  ==  t ][['TRACK_ID','X','Y']]
    for id in data['TRACK_ID'].unique():
        idss.update(id,idxy[ idxy['TRACK_ID'] == id ][['X','Y']].values   )
        idss.draw()
    ### direct version
    """
    #"""
    avid=data[data['OBJECT_TYPE']=='AGENT']['TRACK_ID'].values[0]
    xy=data[data['TIMESTAMP']  ==  t ][['X','Y']].values
    plt.scatter(xy[:,0],xy[:,1])
    idxy=  data[data['TIMESTAMP']  ==  t ][['TRACK_ID','X','Y']].values
    try:
        idxy[idxy['TRACK_ID']==avid][['X','Y']]
        plt.scatter(xy[:,0],xy[:,1],c='red')
    except:
        pass
    #"""

def draw_predictions(file_id):
    max_prob=max(result['forecasted_probabilities'][file_id])
    for i in range(len(result['forecasted_trajectories'][file_id])):
        traj=result['forecasted_trajectories'][file_id][i]
        #if result['forecasted_probabilities'][file_id][i] == max_prob:
        plt.plot(traj[:,0],traj[:,1],'c--')
        #else:
    max_id=result['forecasted_probabilities'][file_id].index(max_prob)
    traj=result['forecasted_trajectories'][file_id][max_id]
    plt.plot(traj[:,0],traj[:,1],'r')


#bigN=1000
#with tqdm(iterable=range(1,bigN),total=bigN-1) as file_id:
with tqdm(iterable=result['forecasted_trajectories'].keys()) as file_id:
    file_id.display(msg='total CSV file')
    n=1
    for fid in file_id:
        data=pd.read_csv(conz.dataset+'val/data/{:}.csv'.format(fid))
        with tqdm(iterable=data['TIMESTAMP'].unique(),total=len(data['TIMESTAMP'].unique())) as ts:
            ts.display(msg='Inner CSV iter')
            idxyss=id_points()
            for t in ts:
                plt.cla()
                draw_map(data,1)
                draw_candidates(data)
                draw_predictions(file_id=fid)
                draw_timed_agents(data,t)

                plt.title(conz.title)
                plt.savefig(conz.result_base_dir+'model_outputs/{:}.jpg'.format(n))
                #plt.show()

                n+=1
            del idxyss
        del data
