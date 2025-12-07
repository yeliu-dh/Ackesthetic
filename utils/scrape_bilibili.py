import requests

def get_bilibili_videos(mid, pages=1):
    """
    获取指定B站博主的视频标题和封面
    :param mid: 博主UID
    :param pages: 获取页数，每页默认30个视频
    :return: 视频列表，每个视频是字典，包含title和cover
    """
    videos = []
    for page in range(1, pages + 1):
        url = "https://api.bilibili.com/x/space/arc/search"
        params = {
            "mid": mid,
            "ps": 30,      # 每页视频数量，最大30
            "pn": page,    # 页码
            "order": "pubdate",
            "jsonp": "jsonp"
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"请求失败，第{page}页")
            continue
        data = response.json()
        vlist = data.get("data", {}).get("list", {}).get("vlist", [])
        for v in vlist:
            videos.append({
                "title": v.get("title"),
                "cover": v.get("pic"),
                "bvid": v.get("bvid")
            })
    return videos

