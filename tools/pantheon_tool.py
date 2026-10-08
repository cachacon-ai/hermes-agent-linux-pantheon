"""Tool schema is enabled only for capability-versioned Pantheon sessions."""
from agent.pantheon_artifacts import SCHEMA, publish
from tools.registry import registry
registry.register(name='pantheon_publish_artifact', toolset='pantheon_artifacts', schema=SCHEMA,
                  handler=lambda args, **kw: publish(args), emoji='📎')
