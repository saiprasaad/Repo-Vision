'''
Goal of Flask Microservice:
1. Flask will take the repository_name such as angular, angular-cli, material-design, D3 from the body of the api sent from React app and 
   will utilize the GitHub API to fetch the created and closed issues. Additionally, it will also fetch the author_name and other 
   information for the created and closed issues.
2. It will use group_by to group the data (created and closed issues) by month and will return the grouped data to client (i.e. React app).
3. It will then use the data obtained from the GitHub API (i.e Repository information from GitHub) and pass it as a input request in the 
   POST body to LSTM microservice to predict and forecast the data.
4. The response obtained from LSTM microservice is also return back to client (i.e. React app).

Use Python/GitHub API to retrieve Issues/Repos information of the past 1 year for the following repositories:
- https: // github.com/angular/angular
- https: // github.com/angular/material
- https: // github.com/angular/angular-cli
- https: // github.com/d3/d3
'''
# Import all the required packages 
import os
import time
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import pandas as pd
import requests

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)

# Add response headers to accept all types of  requests
def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

# Modify response headers when returning to the origin
def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/test', methods=['GET'])
def test():
    # Return a simple response for GET requests
    return jsonify({"message": "Hello, World Sai!"})

@app.route('/api/github/fetch', methods=['POST'])
def fetchForAllRepos():
    body = request.get_json()
    repo_name = body['repository']
    # Add your own GitHub Token to run it local
    token = os.environ.get(
        'GITHUB_TOKEN', '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')
    # token = os.environ.get(
    #     'GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')

    json_all_repos_response = []
    GITHUB_URL = f"https://api.github.com/"
    headers = {
        "Authorization": f'token {token}'
    }
    params = {
        "state": "all"
    }
    for repo_name in body['repository']:
        repository_url = GITHUB_URL + "repos/" + repo_name
        # Fetch GitHub data from GitHub API
        repository = requests.get(repository_url, headers=headers, params=params)
        # Convert the data obtained from GitHub API to JSON format
        repository = repository.json()
        
        today = date.today()

        issues_reponse = []

        # Fetching past 2 months data
        for i in range(2):
            last_month = today + dateutil.relativedelta.relativedelta(months=-1)
            types = 'type:issue'
            repo = 'repo:' + repo_name
            ranges = 'created:' + str(last_month) + '..' + str(today)
            per_page = 'per_page=100'
            search_query = types + ' ' + repo + ' ' + ranges
            query_url = GITHUB_URL + "search/issues?q=" + search_query + "&" + per_page
            # requsets.get will fetch requested query_url from the GitHub API
            search_issues = requests.get(query_url, headers=headers)
            # Convert the data obtained from GitHub API to JSON format
            search_issues = search_issues.json()
            issues_items = []
            try:
                issues_items = search_issues.get("items")
            except KeyError:
                error = {"error": "Data Not Available"}
                resp = Response(json.dumps(error), mimetype='application/json')
                resp.status_code = 500
                return resp
            if issues_items is None:
                today = last_month
                continue
            for issue in issues_items:
                label_name = []
                data = {}
                current_issue = issue
                # Get issue number
                data['issue_number'] = current_issue["number"]
                # Get created date of issue
                data['created_at'] = current_issue["created_at"][0:10]
                if current_issue["closed_at"] == None:
                    data['closed_at'] = current_issue["closed_at"]
                else:
                    # Get closed date of issue
                    data['closed_at'] = current_issue["closed_at"][0:10]
                for label in current_issue["labels"]:
                    # Get label name of issue
                    label_name.append(label["name"])
                data['labels'] = label_name
                # It gives state of issue like closed or open
                data['State'] = current_issue["state"]
                # Get Author of issue
                data['Author'] = current_issue["user"]["login"]
                issues_reponse.append(data)

            today = last_month

        df = pd.DataFrame(issues_reponse)

        '''
        Created Issues is grouped by Month
        ''' 
        created_at_issues = []
        numberOfCreatedIssues = 0
        if "created_at" in df: 
            created_at = df['created_at']
            month_issue_created = pd.to_datetime(
                pd.Series(created_at), format='%Y-%m-%d')
            month_issue_created = month_issue_created.dropna()
            month_issue_created.index = month_issue_created.dt.to_period('m')
            month_issue_created = month_issue_created.groupby(level=0).size()
            if not pd.isna(month_issue_created.index.min()) and not pd.isna(month_issue_created.index.max()):
                month_issue_created = month_issue_created.reindex(pd.period_range(
                    month_issue_created.index.min(), month_issue_created.index.max(), freq='m'), fill_value=0)
            else:
                last_two_months = pd.period_range(end=pd.Period.now(), periods=2, freq='M')
                month_issue_created = month_issue_created.reindex(last_two_months, fill_value=0)
            month_issue_created_dict = month_issue_created.to_dict()
            numberOfCreatedIssues = 0
            for key in month_issue_created_dict.keys():
                count = month_issue_created_dict[key]
                array = [str(key), count]
                created_at_issues.append(array)
                numberOfCreatedIssues = numberOfCreatedIssues + count

        '''
        Closed Issues is grouped by Week
        ''' 

        numberOfClosedIssues = 0
        closed_at_issues = []
        if "closed_at" in df: 
            # Dropping null values
            df = df.dropna(subset=['closed_at'])
            df['closed_at'] = pd.to_datetime(df['closed_at'], errors='coerce')
            # GConverting to week
            df['week_closed'] = df['closed_at'].dt.to_period('W-Mon')
            # Grouping by size
            week_issue_closed = df.groupby('week_closed').size()
            # Grouping
            week_closed = df.groupby('week_closed').size()
            # Creating a dictionary of weekly closed issues
            week_issue_closed_dict = week_issue_closed.to_dict()
            numberOfClosedIssues = 0
            for key in week_issue_closed_dict.keys():
                count = week_issue_closed_dict[key]
                array = [str(key), count]
                closed_at_issues.append(array)
                numberOfClosedIssues = numberOfClosedIssues + count

        # Defining the response
        json_response = {
            "issuesCreated": created_at_issues,
            "issuesClosed": closed_at_issues,
            "numberOfStars": repository["stargazers_count"],
            "numberOfForks": repository["forks_count"],
            "numberOfIssues":[numberOfCreatedIssues, numberOfClosedIssues],
            "repositoryName" : repo_name.split("/")[1]
        }
        json_all_repos_response.append(json_response)
    return jsonify(json_all_repos_response)


@app.route('/api/github', methods=['POST'])
def github():
    body = request.get_json()
    # Extract the choosen repositories from the request
    repo_name = body['repository']
    print(repo_name)
    # Add your own GitHub Token to run it local
    token = os.environ.get(
        'GITHUB_TOKEN', '')
    # token = os.environ.get(
    #     'GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
    GITHUB_URL = f"https://api.github.com/"
    headers = {
        "Authorization": f'token {token}'
    }
    params = {
        "state": "open"
    }
    repository_url = GITHUB_URL + "repos/" + repo_name
    # Fetch GitHub data from GitHub API
    repository = requests.get(repository_url, headers=headers)
    # Convert the data obtained from GitHub API to JSON format
    repository = repository.json()

    today = date.today()

    issues_reponse = []
    # Iterating to get issues for every month for the past 12 months
    for i in range(12):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        types = 'type:issue'
        repo = 'repo:' + repo_name
        ranges = 'created:' + str(last_month) + '..' + str(today)
        # By default GitHub API returns only 30 results per page
        # The maximum number of results per page is 100
        # For more info, visit https://docs.github.com/en/rest/reference/repos 
        per_page = 'per_page=100'
        # Search query will create a query to fetch data for a given repository in a given time range
        search_query = types + ' ' + repo + ' ' + ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "search/issues?q=" + search_query + "&" + per_page
        # requsets.get will fetch requested query_url from the GitHub API
        search_issues = requests.get(query_url, headers=headers, params=params)
        # Convert the data obtained from GitHub API to JSON format
        search_issues = search_issues.json()
        issues_items = []
        try:
            # Extract "items" from search issues
            issues_items = search_issues.get("items")
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if issues_items is None:
            continue
        for issue in issues_items:
            label_name = []
            data = {}
            current_issue = issue
            # Get issue number
            data['issue_number'] = current_issue["number"]
            # Get created date of issue
            data['created_at'] = current_issue["created_at"][0:10]
            if current_issue["closed_at"] == None:
                data['closed_at'] = current_issue["closed_at"]
            else:
                # Get closed date of issue
                data['closed_at'] = current_issue["closed_at"][0:10]
            for label in current_issue["labels"]:
                # Get label name of issue
                label_name.append(label["name"])
            data['labels'] = label_name
            # It gives state of issue like closed or open
            data['State'] = current_issue["state"]
            # Get Author of issue
            data['Author'] = current_issue["user"]["login"]
            issues_reponse.append(data)

        today = last_month

    df = pd.DataFrame(issues_reponse)

    # Daily Created Issues
    df_created_at = df.groupby(['created_at'], as_index=False).count()
    dataFrameCreated = df_created_at[['created_at', 'issue_number']]
    dataFrameCreated.columns = ['date', 'count']

    '''
    Monthly Created Issues
    Format the data by grouping the data by month
    ''' 
    created_at = df['created_at']
    month_issue_created = pd.to_datetime(
        pd.Series(created_at), format='%Y-%m-%d', errors='ignore')
    month_issue_created.index = month_issue_created.dt.to_period('m')
    month_issue_created = month_issue_created.groupby(level=0).size()
    month_issue_created = month_issue_created.reindex(pd.period_range(
        month_issue_created.index.min(), month_issue_created.index.max(), freq='m'), fill_value=0)
    month_issue_created_dict = month_issue_created.to_dict()
    created_at_issues = []
    for key in month_issue_created_dict.keys():
        array = [str(key), month_issue_created_dict[key]]
        created_at_issues.append(array)

    '''
    Monthly Closed Issues
    Format the data by grouping the data by month
    ''' 
    
    closed_at = df['closed_at'].sort_values(ascending=True)
    month_issue_closed = pd.to_datetime(
        pd.Series(closed_at), format='%Y-%m-%d',errors='ignore')
    month_issue_closed.index = month_issue_closed.dt.to_period('m')
    month_issue_closed = month_issue_closed.groupby(level=0).size()
    month_issue_closed = month_issue_closed.reindex(pd.period_range(
        month_issue_closed.index.min(), month_issue_closed.index.max(), freq='m'), fill_value=0)
    month_issue_closed_dict = month_issue_closed.to_dict()
    closed_at_issues = []
    for key in month_issue_closed_dict.keys():
        array = [str(key), month_issue_closed_dict[key]]
        closed_at_issues.append(array)


    today = date.today()

    i = 1

    pulls_response = []
    # Iterating to get issues for every month for the past 12 months
    for i in range(10):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        ranges = 'created:' + str(last_month) + '..' + str(today)
        per_page = 'per_page=100&page='+str(i)
        # Search query will create a query to fetch data for a given repository in a given time range
        search_query = ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "repos/"+repo_name+"/pulls?q=" +  per_page
        # requsets.get will fetch requested query_url from the GitHub API
        search_pulls = requests.get(query_url, headers=headers, params={
        "state": "closed"
        })
        # requsets.get will fetch requested query_url from the GitHub API
        search_pulls = requests.get(query_url, headers=headers)
        # Convert the data obtained from GitHub API to JSON format
        search_pulls = search_pulls.json()
        pulls_items = []
        try:
            # Extract "items" from search issues
            pulls_items = search_pulls
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if issues_items is None:
            continue
        for pull in pulls_items:
            label_name = []
            data = {}
            current_pull = pull
            # Get issue number
            data['issue_number'] = current_pull["number"]
            # Get created date of issue
            data['created_at'] = current_pull["created_at"][0:10]
            year = int(data['created_at'][:4])
            # Checking if the year is less than 2023
            if year < 2023:
                continue
            for label in current_pull["labels"]:
                # Get label name of issue
                label_name.append(label["name"])
            data['labels'] = label_name
            # It gives state of issue like closed or open
            data['State'] = current_pull["state"]
            # Get Author of issue
            data['Author'] = current_pull["user"]["login"]
            pulls_response.append(data)

        today = last_month

    df = pd.DataFrame(pulls_response)


    i = 1

    commits_response = []
    today = date.today()
    for i in range(50):
        params = {
        'since': '2023-05-12T00:00:00Z',
        'until': '2024-04-1223:59:59Z'
        }
        per_page = 'per_page=1000&page='+str(i)
        query_url = GITHUB_URL + "repos/"+repo_name+"/commits?q=" +  per_page
        search_commits = requests.get(query_url, headers=headers, params=params)
        search_commits = search_commits.json()
        commits_items = []
        try:
            commits_items = search_commits
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if commits_items is None:
            continue
        for commit in commits_items:
            label_name = []
            data = {}
            current_commit = commit
            data['issue_number'] = current_commit["sha"]
            data['created_at'] = current_commit["commit"]["committer"]["date"][0:10]
            commits_response.append(data)

        today = last_month
    df = pd.DataFrame(commits_response)
    today = date.today()

    i = 1

    branches_response = []
    # Iterating to get issues for every month for the past 12 months
    for i in range(10):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        ranges = 'created:' + str(last_month) + '..' + str(today)
        per_page = 'per_page=100&page='+str(i)
        # Search query will create a query to fetch data for a given repository in a given time range
        search_query = ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "repos/"+repo_name+"/branches?q=" +  per_page
        # requsets.get will fetch requested query_url from the GitHub API
        search_branches = requests.get(query_url, headers=headers, params={
        "state": "closed"
        })
        # requsets.get will fetch requested query_url from the GitHub API
        search_branches = requests.get(query_url, headers=headers)
        # Convert the data obtained from GitHub API to JSON format
        search_branches = search_branches.json()
        branches_items = []
        try:
            branches_items = search_branches
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if issues_items is None:
            continue
        for branch in branches_items:
            label_name = []
            data = {}
            current_branch = branch
            # Get issue number
            data['issue_number'] = current_branch["commit"]["sha"]
            # Get created date of issue
            query_url = GITHUB_URL + "repos/"+repo_name+"/commits/"+ current_branch["commit"]["sha"]

            result_date = requests.get(query_url,headers=headers)
            # Convert the data obtained from GitHub API to JSON format
            result_date = result_date.json()
            data['created_at'] = result_date["commit"]["committer"]["date"][0:10]
            branches_response.append(data)

        today = last_month

    df = pd.DataFrame(branches_response)

    today = date.today()

    i = 1

    releases_response = []
    for i in range(10):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        ranges = 'created:' + str(last_month) + '..' + str(today)
        per_page = 'per_page=100&page='+str(i)
        # Search query will create a query to fetch data for a given repository in a given time range
        search_query = ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "repos/"+repo_name+"/releases?q=" +  per_page
        # requsets.get will fetch requested query_url from the GitHub API
        search_releases = requests.get(query_url, headers=headers)
        # Convert the data obtained from GitHub API to JSON format
        search_releases = search_releases.json()
        releases_items = []
        try:
            releases_items = search_releases
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if issues_items is None:
            continue
        for release in releases_items:
            label_name = []
            data = {}
            current_release = release
            data['issue_number'] = current_release["id"]
            data['created_at'] = current_release["created_at"][0:10]
            year = int(data['created_at'][:4])
            if year < 2023:
                continue
            releases_response.append(data)

        today = last_month

    df = pd.DataFrame(releases_response)

    today = date.today()

    i = 1

    contributors_response = []
    for i in range(10):
        query_url = GITHUB_URL + "repos/"+repo_name+"/contributors"
        # requsets.get will fetch requested query_url from the GitHub API
        search_contributors = requests.get(query_url, headers=headers)
        # Convert the data obtained from GitHub API to JSON format
        search_contributors = search_contributors.json()
        contributors_items = []
        try:
            contributors_items = search_contributors
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if contributors_items is None:
            continue
        for contributor in contributors_items:
            label_name = []
            data = {}
            current_contributors = contributor
            data['issue_number'] = current_contributors["id"]
            query_url = current_contributors['url'] + "/events"
            search_contributors_by_user_response = requests.get(query_url, headers=headers)
            search_contributors_by_user_response = search_contributors_by_user_response.json()
            for contribution in search_contributors_by_user_response:
                current_contributors_user = contribution
                if(current_contributors_user["repo"]["name"] == repo_name):
                    data['created_at'] = current_contributors_user["created_at"][0:10]
                    year = int(data['created_at'][:4])
                    if year < 2023:
                        continue
                    contributors_response.append(data)

        today = last_month

    df = pd.DataFrame(contributors_response)


    '''
        1. Hit LSTM Microservice by passing issues_response as body
        2. LSTM Microservice will give a list of string containing image paths hosted on google cloud storage
        3. On recieving a valid response from LSTM Microservice, append the above json_response with the response from
            LSTM microservice
    '''
    created_at_body = {
        "issues": issues_reponse,
        "type": "created_at",
        "repo": repo_name.split("/")[1],
        "category": "created_at_issues"
    }
    closed_at_body = {
        "issues": issues_reponse,
        "type": "closed_at",
        "repo": repo_name.split("/")[1],
        "category": "closed_at_issues"
    }
    pulls_body = {
        "issues": pulls_response,
        "type": "created_at",
        "repo": repo_name.split("/")[1],
        "category": "pulls"
    }
    commits_body = {
        "issues": commits_response,
        "type": "created_at",
        "repo": repo_name.split("/")[1],
        "category": "commits"
    }
    branches_body = {
        "issues": branches_response,
        "type": "created_at",
        "repo": repo_name.split("/")[1],
        "category": "branches"
    }
    contributions_body = {
        "issues": contributors_response,
        "type": "created_at",
        "repo": repo_name.split("/")[1],
        "category": "contributions"
    }
    releases_body = {
        "issues": releases_response,
        "type": "created_at",
        "repo": repo_name.split("/")[1],
        "category": "releases"
    }
    max_at_body = {
        "issues": issues_reponse,
        "type": "created_at",
        "type2": "closed_at",
        "repo": repo_name.split("/")[1],
        "category": "created_at_issues"
    }


    # Update your Google cloud deployed LSTM app URL (NOTE: DO NOT REMOVE "/")
    # For local use this:
    LSTM_API_URL = "http://127.0.0.1:8080/" + "api/forecast"
    STATS_API_URL = "http://127.0.0.1:8080/" + "api/forecast/statsmodel"
    PROPHET_API_URL = "http://127.0.0.1:8080/" + "api/forecast/prophet"
    # For dev use this:
    # LSTM_API_URL = "https://lstm-assignment-5-new-eqkc6mfasq-uc.a.run.app/" + "api/forecast"
    # STATS_API_URL = "https://lstm-assignment-5-new-eqkc6mfasq-uc.a.run.app/" + "api/forecast/statsmodel"
    # PROPHET_API_URL = "https://lstm-assignment-5-new-eqkc6mfasq-uc.a.run.app/" + "api/forecast/prophet"

    start_time_lstm = time.time()

    created_at_response = requests.post(LSTM_API_URL,
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})
    
    closed_at_response = requests.post(LSTM_API_URL,
                                       json=closed_at_body,
                                       headers={'content-type': 'application/json'})
    
    pulls_response = requests.post(LSTM_API_URL,
                                       json=pulls_body,
                                       headers={'content-type': 'application/json'})
    
    commits_response = requests.post(LSTM_API_URL,
                                       json=commits_body,
                                       headers={'content-type': 'application/json'})
    
    branches_response = requests.post(LSTM_API_URL,
                                       json=branches_body,
                                       headers={'content-type': 'application/json'})
    
    contributors_response = requests.post(LSTM_API_URL,
                                       json=contributions_body,
                                       headers={'content-type': 'application/json'})
    
    releases_response = requests.post(LSTM_API_URL,
                                       json=releases_body,
                                       headers={'content-type': 'application/json'})

    max_values = requests.post(LSTM_API_URL+"/max",
                                        json=max_at_body,
                                        headers={'content-type': 'application/json'})
    
    end_time_lstm = time.time()

    print("LSTM Totally took:", (end_time_lstm - start_time_lstm), "seconds")

    start_time_stats = time.time()

    stats_created_at_response = requests.post(STATS_API_URL,
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})
    
    stats_closed_at_response = requests.post(STATS_API_URL,
                                    json=closed_at_body,
                                    headers={'content-type': 'application/json'})
    stats_pulls_response = requests.post(STATS_API_URL,
                                    json=pulls_body,
                                    headers={'content-type': 'application/json'})
    stats_commits_response = requests.post(STATS_API_URL,
                                    json=commits_body,
                                    headers={'content-type': 'application/json'})
    stats_branches_response = requests.post(STATS_API_URL,
                                    json=branches_body,
                                    headers={'content-type': 'application/json'})
    stats_contributors_response = requests.post(STATS_API_URL,
                                    json=contributions_body,
                                    headers={'content-type': 'application/json'})
    stats_releases_response = requests.post(STATS_API_URL,
                                    json=releases_body,
                                    headers={'content-type': 'application/json'})
    max_values_stat = requests.post(STATS_API_URL+"/max",
                                        json=max_at_body,
                                        headers={'content-type': 'application/json'})
    
    end_time_stats = time.time()

    print("StatsModel Totally took:", (end_time_stats - start_time_stats), "seconds")

    start_time_prophet = time.time()
    
    prophet_created_at_response = requests.post(PROPHET_API_URL,
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})   
    prophet_closed_at_response = requests.post(PROPHET_API_URL,
                                        json=closed_at_body,
                                        headers={'content-type': 'application/json'})
    prophet_pulls_response = requests.post(PROPHET_API_URL,
                                        json=pulls_body,
                                        headers={'content-type': 'application/json'})
    prophet_commits_response = requests.post(PROPHET_API_URL,
                                        json=commits_body,
                                        headers={'content-type': 'application/json'})
    prophet_branches_response = requests.post(PROPHET_API_URL,
                                        json=branches_body,
                                        headers={'content-type': 'application/json'})
    prophet_contrubutors_response = requests.post(PROPHET_API_URL,
                                        json=contributions_body,
                                        headers={'content-type': 'application/json'})
    prophet_releases_response = requests.post(PROPHET_API_URL,
                                        json=releases_body,
                                        headers={'content-type': 'application/json'})
    max_values_prophet = requests.post(PROPHET_API_URL+"/max",
                                        json=max_at_body,
                                        headers={'content-type': 'application/json'})
    
    end_time_prophet = time.time()

    print("Prophet Totally took:", (end_time_prophet - start_time_prophet), "seconds")


    json_response = prepare_json_response(pulls_response, commits_response, branches_response, releases_response, contributors_response, created_at_response, closed_at_response, max_values, stats_created_at_response, stats_closed_at_response, stats_pulls_response, stats_commits_response, stats_branches_response, stats_contributors_response, stats_releases_response, max_values_stat, prophet_created_at_response, prophet_closed_at_response, prophet_pulls_response, prophet_commits_response, prophet_branches_response, prophet_contrubutors_response, prophet_releases_response, max_values_prophet)

    return jsonify(json_response)


# Prepares the json response to be returned
def prepare_json_response(pulls_response, commits_response, branches_response, releases_response, contributors_response, created_at_response, closed_at_response, max_values, stats_created_at_response, stats_closed_at_response, stats_pulls_response, stats_commits_response, stats_branches_response, stats_contributors_response, stats_releases_response, max_values_stat, prophet_created_at_response, prophet_closed_at_response, prophet_pulls_response, prophet_commits_response, prophet_branches_response, prophet_contrubutors_response, prophet_releases_response, max_values_prophet):
    json_response = {
        "createdAtImageUrls": {
        },
        "closedAtImageUrls": {
        },
        "pullsImageUrls": {
        },
        "commitsImageUrls": {
        },
        "branchesImageUrls": {
        },
        "contributionsImageUrls": {
        },
        "releasesImageUrls": {
        },
        "max_values": {
        },
        "statsCreatedAt" : {
        },
        "statsClosedAt" : {
        },
        "statsPulls" : {
        },
        "statsCommits" : {
        },
        "statsBranches" : {
        },
        "statsContributors" : {
        },
        "statsReleases" : {
        },
        "max_values_stat": {
        },
        "prophetCreatedAt" : {
        },
        "prophetClosedAt" : {
        },
        "prophetPulls" : {
        },
        "prophetCommits" : {
        },
        "prophetBranches" : {
        },
        "prophetContributors" : {
        },
        "prophetReleases" : {
        },
        "max_values_prophet" : {
        }
    }


    if created_at_response and created_at_response.json():
        json_response["createdAtImageUrls"].update(created_at_response.json())
    if closed_at_response and closed_at_response.json():
        json_response["closedAtImageUrls"].update(closed_at_response.json())
    if pulls_response and pulls_response.json():
        json_response["pullsImageUrls"].update(pulls_response.json())
    if commits_response and commits_response.json():
        json_response["commitsImageUrls"].update(commits_response.json())
    if branches_response and branches_response.json():
        json_response["branchesImageUrls"].update(branches_response.json())
    if contributors_response and contributors_response.json():
        json_response["contributionsImageUrls"].update(contributors_response.json())
    if releases_response and releases_response.json():
        json_response["releasesImageUrls"].update(releases_response.json())
    if max_values and max_values.json():
        json_response["max_values"].update(max_values.json())


    if stats_created_at_response and stats_created_at_response.json():
        json_response["statsCreatedAt"].update(stats_created_at_response.json())
    if stats_closed_at_response and stats_closed_at_response.json():
        json_response["statsClosedAt"].update(stats_closed_at_response.json())
    if stats_pulls_response and stats_pulls_response.json():
        json_response["statsPulls"].update(stats_pulls_response.json())
    if stats_commits_response and stats_commits_response.json():
        json_response["statsCommits"].update(stats_commits_response.json())
    if stats_branches_response and stats_branches_response.json():
        json_response["statsBranches"].update(stats_branches_response.json())
    if stats_contributors_response and stats_contributors_response.json():
        json_response["statsContributors"].update(stats_contributors_response.json())
    if stats_releases_response and stats_releases_response.json():
        json_response["statsReleases"].update(stats_releases_response.json())
    if max_values_stat and max_values_stat.json():
        json_response["max_values_stat"].update(max_values_stat.json())


    if prophet_created_at_response and prophet_created_at_response.json():
        json_response["prophetCreatedAt"].update(prophet_created_at_response.json())
    if prophet_closed_at_response and prophet_closed_at_response.json():
        json_response["prophetClosedAt"].update(prophet_closed_at_response.json())
    if prophet_pulls_response and prophet_pulls_response.json():
        json_response["prophetPulls"].update(prophet_pulls_response.json())
    if prophet_commits_response and prophet_commits_response.json():
        json_response["prophetCommits"].update(prophet_commits_response.json())
    if prophet_branches_response and prophet_branches_response.json():
        json_response["prophetBranches"].update(prophet_branches_response.json())
    if prophet_contrubutors_response and prophet_contrubutors_response.json():
        json_response["prophetContributors"].update(prophet_contrubutors_response.json())
    if prophet_releases_response and prophet_releases_response.json():
        json_response["prophetReleases"].update(prophet_releases_response.json())
    if max_values_prophet and max_values_prophet.json():
        json_response["max_values_prophet"].update(max_values_prophet.json())
    return json_response


# Run flask app server on port 5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
