import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import pydst
import ipywidgets as widgets
dst = pydst.Dst(lang='da')
#################################################################################################################################################

#################################################################################################################################################
#Analysing data from MEDICIN4
#creates dictionary for data
columns_dict = {}
columns_dict['AGEBYGROUP'] = 'agegroup'
columns_dict['MEDICINTYPE'] = 'medicinetype'
columns_dict['TID'] = 'year'
columns_dict['INDHOLD'] = 'count'
columns_dict['Bnøgle'] = 'unit'

var_dict = {} # var is for variable
var_dict['N05 Psycholeptica'] = '5'
#var_dict['N06 Psychoanaleptica'] = '6'

unit_dict = {}
unit_dict['Personer'] = 'person'
unit_dict['Indløste recepter'] = 'recepter'

#import data on use of medicine from dst
Medicin4_true = dst.get_data(table_id = 'MEDICIN4', variables={'TID':['*'], 'AGEBYGROUP':['*'], 
'MEDICINTYPE':['*']})

#removes all medicinetypes that isn't Psycholeptica (downer used for anxiety and OCD) and removes the gender classifications
Medicin4_true.rename(columns=columns_dict,inplace=True)
I = (Medicin4_true['medicinetype'] == 'N05 Psycholeptica') & (Medicin4_true['agegroup'] != 'Alder i alt') & ( Medicin4_true['BNØGLE']=='Personer') & (Medicin4_true['KØN'] == 'Køn i alt')
Medicine=Medicin4_true[I]

#changes the count variable to a numerical (integer)
Medicine['count'] = Medicine['count'].astype(int)
Medicine['year'] = Medicine['year'].astype(int)

#drops unnecessary colums
Medicine_drop = Medicine.drop('BNØGLE',axis=1)
#final data frame
Medicine_clean = Medicine_drop.drop('KØN',axis=1)
#################################################################################################################################################

#################################################################################################################################################
#figure 1:
def figure_1():

    # Create pivot table with 'year' and 'agegroup' as indices and 'count' as values
    med_pivot = pd.pivot_table(Medicine_clean, values='count', index=['agegroup'], columns=['year'])

    # Create figure and axis objects
    fig, ax = plt.subplots()

    # Set width of each bar
    bar_width = 0.15

    # Create a bar for each year and age group
    for i, year in enumerate(med_pivot.columns):
        x_pos = np.arange(len(med_pivot.index)) + (i * bar_width)
        ax.bar(x_pos, med_pivot[year], width=bar_width, label=str(year))

    # Set x-axis and y-axis labels and ticks
    ax.set_xticks(np.arange(len(med_pivot.index)))
    ax.set_xticklabels(med_pivot.index, rotation=45, ha='right')
    ax.set_xlabel('Age group')
    ax.set_ylabel('Count')

    # Set legend
    ax.legend(title='Year')

    # Set title
    ax.set_title('Development in use of psycholeptica by age group')


    # Show the plot
    plt.show()

# interactive plot - figur 1
#import ipywidgets as widgets
#from IPython.display import display
def figure_1_int():
    def plot_timeseries(agegroup):
        # Create pivot table with 'year' as index and 'count' as values for selected age group
        med_pivot = Medicine_clean[Medicine_clean['agegroup'] == agegroup].pivot_table(values='count', index=['year'])

        # Create figure and axis objects
        fig, ax = plt.subplots()

        # Plot the timeseries
        ax.plot(med_pivot.index, med_pivot['count'], marker='o')

        # Set x-axis and y-axis labels
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')

        # Set title
        ax.set_title(f'Development in use of psycholeptica for {agegroup} ')

        # Show the plot
        plt.show()

    #Create dropdown menu with age groups
    agegroup_dropdown = widgets.Dropdown(
        options=Medicine_clean['agegroup'].unique(),
        value=Medicine_clean['agegroup'].unique()[0],
        description='Age group:',
        disabled=False,
    )

    interactive_plot = widgets.interactive(plot_timeseries, agegroup=agegroup_dropdown)
    display(interactive_plot)
#################################################################################################################################################

#################################################################################################################################################
#Figure 2 - percentage change
# Group the data by agegroup
Medicine_clean = Medicine_clean.sort_values(['year', 'agegroup'])
Medicin_pct = Medicine_clean.groupby('agegroup')

# Calculate the percentage change in count for each year
growth_rates = Medicin_pct['count'].pct_change()

# Group the data by year and agegroup
Medicin_pct = Medicine_clean.groupby(['agegroup'])

# Calculate the percentage change in count for each group
growth_rates = Medicin_pct['count'].pct_change()

# Add the growth_rate series as a new column to the original dataframe
Medicine_clean['growth_rate'] = growth_rates*100

def figure_2():
    med_pivot = pd.pivot_table(Medicine_clean, values='growth_rate', index=['agegroup'], columns=['year'])
    
    # Create figure and axis objects
    fig, ax = plt.subplots()
    
    # Set width of each bar
    bar_width = 0.15
    
    # Create a bar for each year and age group
    for i, year in enumerate(med_pivot.columns):
        x_pos = np.arange(len(med_pivot.index)) + (i * bar_width)
        ax.bar(x_pos, med_pivot[year], width=bar_width, label=str(year))
    
    # Set x-axis and y-axis labels and ticks
    ax.set_xticks(np.arange(len(med_pivot.index)))
    ax.set_xticklabels(med_pivot.index, rotation=45, ha='right')
    ax.set_xlabel('Age group')
    ax.set_ylabel('Growth rate')
    
    # Set legend
    ax.legend(title='Year')
    
    # Set title
    ax.set_title('Development in use of psycholeptica by age group')
    
    # Show the plot
    plt.show()

#figure 2 interactive
def figure_2_int():
    def plot_timeseries(agegroup):
        # Create pivot table with 'year' as index and 'count' as values for selected age group
        med_pivot = Medicine_clean[Medicine_clean['agegroup'] == agegroup].pivot_table(values='growth_rate', index=['year'])

        # Create figure and axis objects
        fig, ax = plt.subplots()

        # Plot the timeseries
        ax.plot(med_pivot.index, med_pivot['growth_rate'], marker='o')

        # Set x-axis and y-axis labels
        ax.set_xlabel('Year')
        ax.set_ylabel('growth rate')

        # Set title
        ax.set_title(f'Development in use of psycholeptica for {agegroup} ')

        # Show the plot
        plt.show()

    #Create dropdown menu with age groups
    agegroup_dropdown = widgets.Dropdown(
        options=Medicine_clean['agegroup'].unique(),
        value=Medicine_clean['agegroup'].unique()[0],
        description='Age group:',
        disabled=False,
    )

    interactive_plot = widgets.interactive(plot_timeseries, agegroup=agegroup_dropdown)
    display(interactive_plot)
#################################################################################################################################################

#################################################################################################################################################
#Cleans data for HFUDD, and finds the needed variables
def getTotalCounts():
    #imports HFUDD11 from dst
    HFUDD11_true = dst.get_data(table_id = 'HFUDD11', variables={'TID':['*'],'ALDER':['*'], 'HFUDD':['*']})
    #Drops unneeded variables
    HFdropped = HFUDD11_true.drop('BOPOMR',axis=1)
    HFdropped = HFdropped.drop('HERKOMST',axis=1)
    HFdropped = HFdropped.drop('KØN',axis=1)
    I = (HFdropped['ALDER'] == 'Alder i alt')
    HFdropped.drop(HFdropped[I].index,inplace=True)
    I = (HFdropped['HFUDD'].str[3] != ' ')
    HFdropped.drop(HFdropped[I].index,inplace=True)
    I = (HFdropped['TID'].astype(int) < 2016)
    HFdropped.drop(HFdropped[I].index,inplace=True)
    I = (HFdropped['TID'].astype(int) > 2021)
    HFdropped.drop(HFdropped[I].index,inplace=True)
    I = (HFdropped['ALDER'].str[0] == '1')
    HFdropped.drop(HFdropped[I].index,inplace=True)
    I = (HFdropped['HFUDD'] == 'H35 Adgangsgivende uddannelsesforløb')
    HFdropped.drop(HFdropped[I].index,inplace=True)
    I = (HFdropped['HFUDD'] == 'H80 Ph.d. og forskeruddannelser')
    HFdropped.drop(HFdropped[I].index,inplace=True)

    HFdropped['NEWALDER'] = HFdropped['ALDER'].str[0] + "0-" + HFdropped['ALDER'].str[0] + "9 år"
    HFdropped.rename(columns={'INDHOLD':'TotalCount'}, inplace=True)
    HFdropped.rename(columns={'TID':'year'}, inplace=True)
    HFcongregate = HFdropped.groupby(['year', 'HFUDD', "NEWALDER"]).sum().reset_index()
    #Agecongregate = HFdropped.groupby(['year', 'NEWALDER']).sum().reset_index()
    uddannelser = HFdropped['HFUDD'].unique().tolist()
    udannelseslist = ['Grundskole', 'Gymnasiale uddannelser','Erhvervsfaglige uddannelser','Korte videregående uddannelser','Mellemlange videregående uddannelser', 'Bacheloruddannelser', 'Lange videregående uddannelser', 'Uoplyst mv.']

    Uddannelsesdict = dict(zip(udannelseslist,uddannelser))
    return HFcongregate, Uddannelsesdict

    #congregate udgøres af 20-69 årige med forskellige uddannelser. Summer ikke til "Alder i alt", da der er nogen uddannelser der er fjernet (H35, H80, og øvrige)


def getMedicin3(UddList):
    #imports MEDICIN3
    Medicin3_true = dst.get_data(table_id = 'MEDICIN3', variables={'TID':['*'], 'MEDICINTYPE':['*'], 'AGEBYGROUP':['*'], 'UDDANNELSE':['*'], })
    #removes unnecessary columns
    Medicin3_dropped = Medicin3_true.drop('BNØGLE',axis=1)

    #cleans data so only the necessary variables are left (people aged 20-69 and only psycholeptica)
    I = (Medicin3_dropped['MEDICINTYPE'] != "N05 Psycholeptica")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    I = (Medicin3_dropped['UDDANNELSE'] == "Uddannelser i alt")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    I = (Medicin3_dropped['AGEBYGROUP'] == "Alder i alt")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    I = (Medicin3_dropped['AGEBYGROUP'].str[0] == "0")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    I = (Medicin3_dropped['AGEBYGROUP'].str[0] == "1")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    I = (Medicin3_dropped['AGEBYGROUP'].str[0] == "7")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    I = (Medicin3_dropped['AGEBYGROUP'].str[0] == "8")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    I = (Medicin3_dropped['AGEBYGROUP'].str[0] == "9")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    I = (Medicin3_dropped['INDHOLD'].str[0] == ".")
    Medicin3_dropped.drop(Medicin3_dropped[I].index,inplace=True)
    Medicin3_dropped["INDHOLD"] = Medicin3_dropped["INDHOLD"].astype(int)

    #renames columns
    Medicin3_dropped.rename(columns={'UDDANNELSE':'HFUDD'}, inplace=True)
    Medicin3_dropped.rename(columns={'INDHOLD':'Count'}, inplace=True)
    Medicin3_dropped.rename(columns={'AGEBYGROUP':'NEWALDER'}, inplace=True)
    Medicin3_dropped.rename(columns={'TID':'year'}, inplace=True)

    #summarizes amount of people by agegroup, year and education level
    for key,value in UddList.items():
        Medicin3_dropped.HFUDD.replace(key,value,inplace=True)
    Medicin3_returnable = Medicin3_dropped.groupby(['year', 'HFUDD', 'NEWALDER']).sum().reset_index()
    return Medicin3_returnable



Totals = getTotalCounts()
Total = Totals[0] #Dataframe of Amount of people with educational level
UDD_List = Totals[1] #List of educations for comparative purposes, to be fed to Medicin3 (which also has educational levels, with slightly different names) 


Medicin3_true =getMedicin3(UDD_List)


#merging medicin3 with Total on agegroup, year and education level
Medicin3_merged = Medicin3_true.merge(Total,how='left', on=['year','HFUDD',"NEWALDER"])

#calculates the share of people in an age group and education level using psycholeptica
ratio=Medicin3_merged['Count']/Medicin3_merged['TotalCount']
Medicin3_merged['Percent']=ratio*100
#################################################################################################################################################

#################################################################################################################################################
#figure 3 interactive plot
def figure_3_int():
    def plot_timeseries(agegroup, edu):
        # Create pivot table with 'year' as index and 'count' as values for selected age group
        merge_pivot = Medicin3_merged[(Medicin3_merged['NEWALDER'] == agegroup) & (Medicin3_merged['HFUDD'] == edu)].pivot_table(values='Percent', index=['year'])

        # Create figure and axis objects
        fig, ax = plt.subplots()

        # Plot the timeseries
        ax.plot(merge_pivot.index, merge_pivot['Percent'], marker='o')

        # Set x-axis and y-axis labels
        ax.set_xlabel('Year')
        ax.set_ylabel('Percent')

        # Set title
        ax.set_title(f'Share of people in {agegroup} with {edu} using psycholeptica')

        # Show the plot
        plt.show()

    #Create dropdown menu with age groups
    agegroup_dropdown = widgets.Dropdown(
        options=Medicin3_merged['NEWALDER'].unique(), #creates an option for each unique value of NEWALDER
        value=Medicin3_merged['NEWALDER'].unique()[0], #sets the default value to the first value we meet
        description='Age group:',
        disabled=False,
    )
    #create dropdown menu with highest achieved education
    edu_dropdown = widgets.Dropdown(
        options=Medicin3_merged['HFUDD'].unique(), #creates an option for each unique value of HFUDD
        value=Medicin3_merged['HFUDD'].unique()[0], #sets the default value to the first value we meet
        description='Education level:',
        disabled=False,
    )

    interactive_plot = widgets.interactive(plot_timeseries, agegroup=agegroup_dropdown, edu=edu_dropdown)
    display(interactive_plot) #"calls" the plot
#################################################################################################################################################

#################################################################################################################################################
# Create a pivot table to group the data by year and age group
def figure_4():
    Medicin3_merged_2029 = Medicin3_merged[Medicin3_merged['NEWALDER']=="20-29 år"]
    pivot = Medicin3_merged_2029.pivot(index='year', columns='HFUDD', values='Percent')
    # Plot the data as a line chart
    pivot.plot(kind='line')

    plt.title('Drug Usage by Educational Level, 2016-2021')
    plt.xlabel('Year')
    plt.ylabel('Percentage of population aged 20-29')
    plt.show()

#################################################################################################################################################

#################################################################################################################################################
# Group the data by agegroup
def figure_5():
    I = (Medicin3_merged['NEWALDER']== "20-29 år")
    med_udd_2029=Medicin3_merged[I]
    med_udd_2029_sort = med_udd_2029.sort_values(['year', 'HFUDD'])
    med_udd_2029_group = med_udd_2029_sort.groupby('HFUDD')
    # Calculate the percentage change in count for each year
    growth_rates_2029 = med_udd_2029_group['Percent'].pct_change()
    # Add the growth_rate series as a new column to the original dataframe
    med_udd_2029['growth_rate'] = growth_rates_2029*100
    
    
    med2029_pivot = pd.pivot_table(med_udd_2029, values='growth_rate', index=['HFUDD'], columns=['year'])
    
    # Create figure and axis objects
    fig, ax = plt.subplots()
    
    # Set width of each bar
    bar_width = 0.1
    
    # Create a bar for each year and age group
    for i, hfuud in enumerate(med2029_pivot.index):
        x_pos = np.arange(len(med2029_pivot.columns)) + (i * bar_width)
        ax.bar(x_pos, med2029_pivot.loc[hfuud], width=bar_width, label=str(hfuud))
    
    # Set x-axis and y-axis labels and ticks
    ax.set_xticks(np.arange(len(med2029_pivot.columns)))
    ax.set_xticklabels(med2029_pivot.columns)
    ax.set_xlabel('Year')
    ax.set_ylabel('Growth rate')
    
    # Set legend
    ax.legend(title='HFUDD', fontsize="2")
    
    # Set title
    ax.set_title('Development in use of psycholeptica by education level')
    
    # Show the plot
    plt.show()

    med_udd_2029_avg = med_udd_2029_sort.groupby('HFUDD')
    # Calculate the percentage change in count for each year
    avg_2029 = med_udd_2029_avg['Percent'].mean()

    print(avg_2029)




