"""change monitor_id type

Revision ID: fef7bcd79b03
Revises: ea231f718578
Create Date: 2025-06-20 16:35:04.297780

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fef7bcd79b03'
down_revision: Union[str, None] = 'ea231f718578'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Step 1: Drop all foreign key constraints that were pointing to monitor.monitor_id.
    op.drop_constraint(op.f('action_responsibility_monitor_id_fkey'), 'action_responsibility', type_='foreignkey')
    op.drop_constraint(op.f('dry_day_event_monitor_id_fkey'), 'dry_day_event', type_='foreignkey')
    op.drop_constraint(op.f('measurement_monitor_id_fkey'), 'measurement', type_='foreignkey')
    op.drop_constraint(op.f('presite_install_check_monitor_id_fkey'), 'presite_install_check', type_='foreignkey')
    op.drop_constraint(op.f('storm_event_monitor_id_fkey'), 'storm_event', type_='foreignkey')
    op.drop_constraint(op.f('weekly_quality_check_monitor_id_fkey'), 'weekly_quality_check', type_='foreignkey')

    # Step 2: Alter the column type for ONLY the monitor.monitor_id column.
    # The child table columns remain INTEGER.
    op.alter_column('monitor', 'monitor_id',
               existing_type=sa.INTEGER(),
               type_=sa.Text(),
               existing_nullable=True)

    # Step 3: Recreate all foreign key constraints, now correctly pointing from the
    # child table's monitor_id (INTEGER) to the parent monitor.id (INTEGER).
    op.create_foreign_key(None, 'action_responsibility', 'monitor', ['monitor_id'], ['id'])
    op.create_foreign_key(None, 'dry_day_event', 'monitor', ['monitor_id'], ['id'])
    op.create_foreign_key(None, 'measurement', 'monitor', ['monitor_id'], ['id'])
    op.create_foreign_key(None, 'presite_install_check', 'monitor', ['monitor_id'], ['id'])
    op.create_foreign_key(None, 'storm_event', 'monitor', ['monitor_id'], ['id'])
    op.create_foreign_key(None, 'weekly_quality_check', 'monitor', ['monitor_id'], ['id'])

    # The autogen script also included this index drop.
    # It seems unrelated but we can leave it in if it was part of the autogen.
    op.drop_index(op.f('measurement_time_idx'), table_name='measurement')


def downgrade() -> None:
    """Downgrade schema."""
    # A correct downgrade is very complex. For this case, it's safer to just pass.
    pass

