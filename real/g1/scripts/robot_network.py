"""Configure Cyclone DDS for the selected G1 wired interface and peer."""
import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

LOCAL_CONFIG = Path(__file__).resolve().parents[1] / "config/local.json"


def dds_config(config, *, peer_key="robot_host"):
    interface = config.get("robot_interface")
    peer = config.get(peer_key)
    for key, value in (("robot_interface", interface), (peer_key, peer)):
        if not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_"):
            raise ValueError(f"Set {key} in the local G1 configuration before opening DDS")
    root = Element("CycloneDDS")
    domain = SubElement(root, "Domain", Id="0")
    interfaces = SubElement(SubElement(domain, "General"), "Interfaces")
    SubElement(interfaces, "NetworkInterface", name=interface)
    peers = SubElement(SubElement(domain, "Discovery"), "Peers")
    SubElement(peers, "Peer", Address=peer)
    return tostring(root, encoding="unicode")


def local_dds_config(path=LOCAL_CONFIG, *, peer_key="robot_host"):
    return dds_config(json.loads(path.read_text()), peer_key=peer_key)
