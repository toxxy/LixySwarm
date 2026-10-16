"""Locally selected executable profiles; downloaded metadata cannot select one."""
PROFILES={
    'own-cpu-programs':{'name':'LixySwarmEngineCPU','operations':['reference.inference.v1','training.gradient.v1','reference.candidate.v1','swarm.gradient.v1','swarm.candidate.v1','swarm.audit.v1','swarm.gradient.v2','swarm.candidate.v2','swarm.validate.v1','swarm.gradient.v3','swarm.candidate.v3'],
        'max_package_bytes':2*1024**3,'max_download_bytes':512*1024**2,'archive_links':False},
    'own-cpu-bootstrap':{'name':'LixySwarmEngineCPU','operations':['reference.inference.v1','training.gradient.v1','reference.candidate.v1','swarm.gradient.v1','swarm.candidate.v1','swarm.audit.v1','swarm.gradient.v2','swarm.candidate.v2','swarm.validate.v1'],
        'max_package_bytes':2*1024**3,'max_download_bytes':512*1024**2,'archive_links':False},
    'own-cpu-network':{'name':'LixySwarmEngineCPU','operations':['reference.inference.v1','training.gradient.v1','reference.candidate.v1','swarm.gradient.v1','swarm.candidate.v1','swarm.audit.v1','swarm.gradient.v2','swarm.candidate.v2'],
        'max_package_bytes':2*1024**3,'max_download_bytes':512*1024**2,'archive_links':False},
    'own-cpu':{'name':'LixySwarmEngineCPU','operations':['reference.inference.v1','training.gradient.v1','reference.candidate.v1','swarm.gradient.v1','swarm.candidate.v1'],
        'max_package_bytes':2*1024**3,'max_download_bytes':512*1024**2,'archive_links':False},
    'own-cpu-continuous':{'name':'LixySwarmEngineCPU','operations':['reference.inference.v1','training.gradient.v1','reference.candidate.v1','swarm.gradient.v1','swarm.candidate.v1','swarm.audit.v1'],
        'max_package_bytes':2*1024**3,'max_download_bytes':512*1024**2,'archive_links':False},
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
    'language-cuda-half':{'name':'LixySwarmLanguageCUDA','operations':['language.inference.v1','language.gradient.v2','language.candidate.v1','language.gradient.v3','language.candidate.v2','language.gradient.v4','language.candidate.v3'],
        'max_package_bytes':8*1024**3,'max_download_bytes':4*1024**3,'archive_links':True},
    'language-cuda-math':{'name':'LixySwarmLanguageCUDA','operations':['language.inference.v1','language.gradient.v2','language.candidate.v1','language.gradient.v3','language.candidate.v2','language.gradient.v4','language.candidate.v3','language.gradient.v5','language.candidate.v4'],
        'max_package_bytes':8*1024**3,'max_download_bytes':4*1024**3,'archive_links':True},
}


def engine_profile(name):
    if name not in PROFILES:raise ValueError('Unknown locally selected engine profile')
    return PROFILES[name]
