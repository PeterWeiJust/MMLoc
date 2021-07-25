
#include<stdio.h>
#include<windows.h>
#include<fstream>        
#include<sstream>        
#include<list>
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<cstdlib>
using namespace std;
void SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}
double caldistance(double wifix, double wifiy, double sensorx, double sensory) {
	return sqrt(pow(wifix - sensorx, 2) + pow(wifiy - sensory, 2));
}

string wifiroute = "D:/src/MM-Loc/ScenarioA/wifi";
string sensorroute = "D:/src/MM-Loc/ScenarioA/sensordata/overlap_timestep1000/";
string outputroute = "D:/src/MM-Loc/ScenarioA/sensor_wifi_timestep1000_";

int main() {
	string value;
	ifstream finwifi, finsensor;

	for (int k = 1; k < 15; k++) {
		finwifi.open(wifiroute + to_string(k) + ".csv", ios::in);
		vector<vector<string>>wifidata;
		while (finwifi.good())
		{
			getline(finwifi, value);
			vector<string>wifirow;
			SplitString(value, wifirow, ",");

			wifidata.push_back(wifirow);
		}
		finwifi.close();
		finwifi.clear();


		finsensor.open(sensorroute + to_string(k) + "_timestep1000_overlap.csv", ios::in);
		vector<vector<string>>sensordata;
		while (finsensor.good())
		{
			getline(finsensor, value);
			vector<string>wifirow;
			SplitString(value, wifirow, ",");
			sensordata.push_back(wifirow);
		}
		finsensor.close();
		finsensor.clear();

		for (int i = 1; i < sensordata.size() - 1; i++) {
			double sensorx = atof(sensordata[i][1].c_str()), sensory = atof(sensordata[i][2].c_str());
			double mindistance = 10000000;
			int minindex = -1;
			for (int j = 1; j < wifidata.size() - 1; j++) {
				double wifix = atof(wifidata[j][0].c_str()), wifiy = atof(wifidata[j][1].c_str());
				double dis = caldistance(wifix, wifiy, sensorx, sensory);
				if (dis < mindistance && atoi(wifidata[j][104].c_str()) < 9 ) {
					mindistance = dis;
					minindex = j;
				}
			}
			if (mindistance <= 0.05 && minindex != -1 && minindex != 0) {
				wifidata[minindex][104] = to_string(atoi(wifidata[minindex][104].c_str()) + 1);
				sensordata[i].insert(sensordata[i].end(), wifidata[minindex].begin() + 2, wifidata[minindex].end() - 1);
				sensordata[i].push_back(to_string(mindistance));
			}
			else {
				for (int k = 0; k < 102; k++) {
					sensordata[i].push_back("0");
				}
				sensordata[i].push_back("-1");
			}

		}

		ofstream out(outputroute + to_string(k) + ".csv", ios::out);
		string columindex = "st,lat,lng,";
		for (int j = 0; j < 102; j++) {

			columindex += to_string(j) + ",";

		}
		columindex += "mind";
		out << columindex << endl;
		for (int i = 1; i < sensordata.size(); ++i)
		{
			string rawdata = "";

			for (int j = 0; j < sensordata[i].size(); j++) {
				if (j == sensordata[i].size() - 1) {
					rawdata += sensordata[i][j];
				}
				else {
					rawdata += sensordata[i][j] + ",";
				}
			}

			out << rawdata << endl;
		}
	}

	return 0;
}