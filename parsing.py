import json
import vk_api
from config import token

def get_comments(group_name, count_comments, number_post):

    # коомментарии с поста
    comments_array = []

    session = vk_api.VkApi(token=token)

    # узнаем id группы, используя короткое имя группы
    info_group = session.method("utils.resolveScreenName", {"screen_name":group_name})
    id_group = info_group["object_id"]

    # сохраняем в файл {group_name}_posts.json все посты
    posts = session.method("wall.get", {"owner_id": -1 * id_group,"count":number_post})
    with open(f"{group_name}_posts.json", "w", encoding="utf-8") as file:
        json.dump(posts, file, indent=4, ensure_ascii=False)

    #узнаем id поста из группы
    number_post = posts["items"][number_post-1]["id"]

    # сохраняем в файл {group_name}_comments.json информацию о комментариях
    with open(f"{group_name}_comments.json", "w", encoding="utf-8") as file:
        comments_dict = session.method("wall.getComments", {"owner_id": -1 * id_group, "post_id": number_post, "count": count_comments})
        json.dump(comments_dict, file, indent=4, ensure_ascii=False)
        for i in range(0,count_comments):
            if len(comments_dict["items"][i]["text"]) > 0:
                comments_array.append(comments_dict["items"][i]["text"])
    return comments_array

