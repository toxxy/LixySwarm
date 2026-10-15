"""Locally selected executable profiles; downloaded metadata cannot select one."""
PROFILES={
    'reference-cpu':{'name':'LixySwarmEngineCPU','operations':['reference.inference.v1','training.gradient.v1','reference.candidate.v1'],
        'max_package_bytes':2*1024**3,'max_download_bytes':512*1024**2,'archive_links':False},
    'language-cpu':{'name':'LixySwarmLanguageCPU','operations':['language.inference.v1'],
        'max_package_bytes':2*1024**3,'max_download_bytes':512*1024**2,'archive_links':True},
    'language-cuda':{'name':'LixySwarmLanguageCUDA','operations':['language.inference.v1'],
        'max_package_bytes':8*1024**3,'max_download_bytes':4*1024**3,'archive_links':True},
    'language-cuda-learning':{'name':'LixySwarmLanguageCUDA','operations':['language.inference.v1','language.gradient.v2','language.candidate.v1'],
        'max_package_bytes':8*1024**3,'max_download_bytes':4*1024**3,'archive_links':True},
    'language-cuda-batches':{'name':'LixySwarmLanguageCUDA','operations':['language.inference.v1','language.gradient.v2','language.candidate.v1','language.gradient.v3','language.candidate.v2'],
        'max_package_bytes':8*1024**3,'max_download_bytes':4*1024**3,'archive_links':True},
}


def engine_profile(name):
    if name not in PROFILES:raise ValueError('Unknown locally selected engine profile')
    return PROFILES[name]
