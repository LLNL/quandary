#include "config.hpp"
using namespace std;


template <typename T>
void export_param(int mpi_rank, std::stringstream& log, string key, T value)
{
  if (mpi_rank == 0)
  {
    log << key << " = " << value << endl;
  }
}

MapParam::MapParam() { 
    mpi_rank = 0; 
}

MapParam::MapParam(MPI_Comm comm_, stringstream& logstream)
{
  comm = comm_;
  MPI_Comm_rank(comm, &mpi_rank);
  log = &logstream;
}

MapParam::~MapParam(){}

void StringTrim(string &s)
{
  string s2(s);
  int nb = 0;
  for (size_t i = 0; i < s2.size(); i++)
    if ((s2[i] != ' ') && (s2[i] != '\t'))
      nb++;

  s.resize(nb);
  nb = 0;
  for (size_t i = 0; i < s2.size(); i++)
    if ((s2[i] != ' ') && (s2[i] != '\t'))
      s[nb++] = s2[i];
}

void MapParam::ReadFile(string filename)
{
  string line;
  ifstream file;
  file.open(filename.c_str());
  if (!file.is_open())
  {
    if (mpi_rank == 0)
      cerr << "Unable to read the file " << filename << endl;
    abort();
  }
  while (getline(file, line))
  {
    StringTrim(line);
    if (line.size() > 0 && line[0] != '#')
    {
      int pos = line.find('=');
      string key = line.substr(0, pos);
      string value = line.substr(pos + 1);
      map<string, string>::iterator it_value = this->find(key);
      if (it_value == this->end())
      {
        this->insert(pair<string, string>(key, value));
      }
      else
      {
        if (mpi_rank == 0)
          cerr << "# Warning: existing param found : " << key << ", with new value " << value << ". Replacing" << endl;
        it_value->second = value;
      }
    }
  }
  file.close();
}

void MapParam::GetVecDoubleParam(string key, vector<double> &fillme, double default_val, bool exportme) const 
{
  map<string, string>::const_iterator it_value = this->find(key);
  if (it_value == this->end())
  {
    if (mpi_rank == 0)
      cerr << "# Warning: parameter " << key << " not found ! Taking default = " << default_val << endl;
      fillme.push_back(default_val);
  }
  else 
  {
    /* Parse the string line w.r.t. comma separator */
    stringstream line(it_value->second); 
    string intermediate; 
    while(getline(line, intermediate, ',')) 
    { 
        fillme.push_back(atof(intermediate.c_str()));
    } 
    if (exportme) export_param(mpi_rank, *log, key, line.str());
  }
}

double MapParam::GetDoubleParam(string key, double default_val) const
{
  map<string, string>::const_iterator it_value = this->find(key);
  double val;
  if (it_value == this->end())
  {
    if (mpi_rank == 0)
      cerr << "# Warning: parameter " << key << " not found ! Taking default = " << default_val << endl;
    val = default_val;
  }
  else
    val = atof(it_value->second.c_str());

  export_param(mpi_rank, *log, key, val);
  return val;
}

int MapParam::GetIntParam(string key, int default_val) const
{
  map<string, string>::const_iterator it_value = this->find(key);
  int val;
  if (it_value == this->end())
  {
    if (mpi_rank == 0)
      cerr << "# Warning: parameter " << key << " not found ! Taking default = " << default_val << endl;
    val = default_val;
  }
  else
    val = atoi(it_value->second.c_str());

  export_param(mpi_rank, *log, key, val);
  return val;
}

string MapParam::GetStrParam(string key, string default_val) const
{
  map<string, string>::const_iterator it_value = this->find(key);
  string val;
  if (it_value == this->end())
  {
    if (mpi_rank == 0)
      cerr << "# Warning: parameter " << key << " not found ! Taking default = " << default_val << endl;
    val = default_val;
  }
  else
    val = it_value->second;

  export_param(mpi_rank, *log, key, val);
  return val;
}

bool MapParam::GetBoolParam(string key, bool default_val) const
{
  map<string, string>::const_iterator it_value = this->find(key);
  bool val;
  if (it_value == this->end())
  {
    if (mpi_rank == 0)
      cerr << "# Warning: parameter " << key << " not found ! Taking default = " << default_val << endl;
    val = default_val;
  }
  else if (!strcmp(it_value->second.c_str(), "yes") || !strcmp(it_value->second.c_str(), "true") || !strcmp(it_value->second.c_str(), "YES") || !strcmp(it_value->second.c_str(), "1"))
    val = true;
  else
    val = false;

  export_param(mpi_rank, *log, key, val);
  return val;
}


int MapParam::GetMpiRank() const { 
    return mpi_rank; 
}


void MapParam::GetVecIntParam(std::string key, std::vector<int> &fillme, int default_val) const 
{
  map<string, string>::const_iterator it_value = this->find(key);
  string lineexp;
  double val;
  if (it_value == this->end())
  {
    if (mpi_rank == 0)
      cerr << "# Warning: parameter " << key << " not found ! Taking default = " << default_val << endl;
      fillme.push_back(default_val);
  }
  else 
  {
    /* Parse the string line w.r.t. comma separator */
    string intermediate; 
    stringstream line(it_value->second); 
    while(getline(line, intermediate, ',')) 
    { 
        fillme.push_back(atoi(intermediate.c_str()));
    } 
    lineexp = line.str();
    export_param(mpi_rank, *log, key, lineexp);  
  }
}


void MapParam::GetVecStrParam(std::string key, std::vector<std::string> &fillme, std::string default_val, bool exportme) const
{
  map<string, string>::const_iterator it_value = this->find(key);
  string lineexp;
  if (it_value == this->end())
  {
    if (mpi_rank == 0)
      cerr << "# Warning: parameter " << key << " not found ! Taking default = " << default_val << endl;
      fillme.push_back(default_val);
  }
  else 
  {
    /* Parse the string line w.r.t. comma separator */
    string intermediate; 
    stringstream line(it_value->second); 
    while(getline(line, intermediate, ',')) 
    { 
        fillme.push_back(intermediate);
    } 
    lineexp = line.str();
    if (exportme) export_param(mpi_rank, *log, key, lineexp);  
  }
}