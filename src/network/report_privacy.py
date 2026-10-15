"""Public experiment routes preserve Internet evidence without operator IPs."""
from copy import deepcopy
import ipaddress


def public_route(route):
    result={key:deepcopy(value) for key,value in route.items() if key not in {'host','relay_host'}}
    address=route.get('relay_host') or route.get('host')
    if address:
        try:result['public_internet']=ipaddress.ip_address(address).is_global
        except ValueError:raise ValueError('Public route evidence requires a literal observed address')
    return result


def redact_report_routes(value, *, protected=False):
    """Never edit a receipt-bound result or a publisher-signed object.

    Routes collected beside those objects are observational metadata and can
    omit their addresses. If an address is inside a signed object, publication
    must fix its producer instead of silently invalidating the signature.
    """
    if isinstance(value,list):return [redact_report_routes(v,protected=protected) for v in value]
    if not isinstance(value,dict):return deepcopy(value)
    protected=(protected or any(key in value for key in ['receipt','signature','signatures'])
        or {'job_id','operation','payload','origin_node_id'}<=set(value))
    route=('transport' in value and 'encrypted' in value and ('host' in value or 'relay_host' in value))
    if route and protected:raise ValueError('Cannot redact receipt-bound or signed route data')
    source=public_route(value) if route else value
    return {key:redact_report_routes(v,protected=protected) for key,v in source.items()}
